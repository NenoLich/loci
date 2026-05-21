use ahash::AHashMap;
use candle_core::cuda::cudarc::types;
use tokenizers::pre_tokenizers::sequence::Sequence;
use std::io::Write;
use std::collections::BTreeSet;

use crate::error::LociError;
use crate::gguf::{GgufInfo, GgufKVMeta};
use crate::api::types::{ChatMessage, Tool};
use crate::config::TokenizerConfig;

use tokenizers::Encoding;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::{byte_level::ByteLevel, PreTokenizerWrapper, split::Split};
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{AddedToken, Tokenizer};
use once_cell::sync::OnceCell;
use minijinja::{Environment, context};

use tempfile::NamedTempFile;

pub struct TokenizerDefaults;

impl TokenizerDefaults {
    pub const CHAT_TEMPLATE: &'static str = "{% for message in messages %}{{ message.content }}{% endfor %}";
    pub const BOS_TOKEN_ID: u32 = 1;
    pub const EOS_TOKEN_ID: u32 = 2;
    pub const UNKNOWN_TOKEN_ID: u32 = 0;
    pub const PADDING_TOKEN_ID: u32 = 2;
    pub const EOT_TOKEN_ID: u32 = 2;
    pub const EOM_TOKEN_ID: u32 = 2;
}

#[derive(Default)]
pub struct StreamState {
    pub tokens: Vec<u32>,
    pub prev_index: usize,
    pub read_index: usize,
}

impl StreamState {
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.read_index = 0;
    }
}

pub struct TokenizerService {
    tokenizer: Tokenizer,
    chat_template: String,
    bos_token_id: u32,
    eos_token_id: u32,
    eot_token_id: u32,
    eom_token_id: u32,
    bos_token: OnceCell<String>,
    eos_token: OnceCell<String>,
    eot_token: OnceCell<String>,
    eom_token: OnceCell<String>,
}

impl TokenizerService {
    pub fn builder() -> TokenizerServiceBuilder {
        TokenizerServiceBuilder::new()
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding, LociError> {
        self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| LociError::Tokenization { source: e })
    }

    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String, LociError> {
        self.tokenizer
            .decode(tokens, skip_special_tokens)
            .map_err(|e| LociError::Tokenization { source: e })
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn process_token(&self, state: &mut StreamState, token: u32) -> Result<Option<String>, LociError> {
        let prev_text = self.decode(&state.tokens[state.prev_index..state.read_index], true)?;
        
        state.tokens.push(token);
        let text = self.decode(&state.tokens[state.prev_index..], true)?;

        if text.len() > prev_text.len() && !text.ends_with('\u{FFFD}') {
            let text = text.split_at(prev_text.len()).1.to_string();
            state.prev_index = state.read_index;
            state.read_index = state.tokens.len();
            Ok(Some(text))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self, state: &mut StreamState) -> Result<Option<String>, LociError> {
        let prev_text = self.decode(&state.tokens[state.prev_index..state.read_index], true)?;
        let text = self.decode(&state.tokens[state.prev_index..], true)?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len()).1.to_string();
            Ok(Some(text))
        } else {
            Ok(None)
        }
    }

    pub fn apply_chat_template(&self, messages: &[ChatMessage], tools: &[Tool]) -> Result<String, LociError> {
        let mut env = Environment::new();
        let name = "chat";
        env.add_template(name, &self.chat_template)
            .map_err(|e| LociError::Tokenization { source: Box::new(e) })?;
        let template = env.get_template(name)
            .map_err(|e| LociError::Tokenization { source: Box::new(e) })?;
        let bos_token = 
            self.bos_token.get_or_try_init(|| self.decode(&[self.bos_token_id], false))?;
        let eos_token = 
            self.eos_token.get_or_try_init(|| self.decode(&[self.eos_token_id], false))?;
        let eot_token = 
            self.eot_token.get_or_try_init(|| self.decode(&[self.eot_token_id], false))?;
        let eom_token = 
            self.eom_token.get_or_try_init(|| self.decode(&[self.eom_token_id], false))?;

        let rendered = template.render(context! {
            bos_token => bos_token,
            eos_token => eos_token,
            eot_token => eot_token,
            eom_token => eom_token,
            keep_past_thinking => false,
            messages => messages,
            tools => tools,
            add_generation_prompt => true,
        }).map_err(|e| LociError::Tokenization { source: Box::new(e) })?;

        Ok(rendered)
    }
}

pub struct TokenizerServiceBuilder {
    chat_template: String,
    bos_token_id: u32,
    eos_token_id: u32,
    eot_token_id: u32,
    eom_token_id: u32,
    config: Option<TokenizerConfig>,
}

impl TokenizerServiceBuilder {
    pub fn new() -> Self {
        Self {
            chat_template: TokenizerDefaults::CHAT_TEMPLATE.into(),
            bos_token_id: TokenizerDefaults::BOS_TOKEN_ID,
            eos_token_id: TokenizerDefaults::EOS_TOKEN_ID,
            eot_token_id: TokenizerDefaults::EOT_TOKEN_ID,
            eom_token_id: TokenizerDefaults::EOM_TOKEN_ID,
            config: None,
        }
    }

    pub fn with_gguf_metadata(mut self, info: &GgufInfo) -> Self {
        let metadata = info.kv_meta.as_slice();
        self.config = Some(TokenizerConfig::from(metadata));
        self
    }

    pub fn build(&mut self) -> Result<TokenizerService, LociError> {
        if self.config.is_none() {
            return Err(LociError::TokenizerBuild {
                reason: "TokenizerConfig is required to build tokenizer but was not set".into(),
            });
        }

        let config = self.config.as_ref().unwrap();
        if config.model_type.is_none() {
            return Err(LociError::TokenizerBuild {
                reason: "model_type is required to build tokenizer but was not found in config".into(),
            });
        }

        if let Some(ref ct) = config.chat_template { self.chat_template = ct.into(); }
        if let Some(id) = config.bos_token_id { self.bos_token_id = id; }
        if let Some(id) = config.eos_token_id { self.eos_token_id = id; }
        if let Some(id) = config.eot_token_id { self.eot_token_id = id; }
        if let Some(id) = config.eom_token_id { self.eom_token_id = id; }


        let tokenizer = if let Some(ref json_config) = config.json_config {
            self.tokenizer_from_json_key(json_config)?
        } else {
            self.tokenizer_from_config(config)?
        };

        Ok(TokenizerService {
            tokenizer,
            chat_template: self.chat_template.clone(),
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            eot_token_id: self.eot_token_id,
            eom_token_id: self.eom_token_id,
            bos_token: OnceCell::new(),
            eos_token: OnceCell::new(),
            eot_token: OnceCell::new(),
            eom_token: OnceCell::new(),
        })
    }

    fn tokenizer_from_json_key(&self, json_config: &str) -> Result<Tokenizer, LociError> {
        let mut file = NamedTempFile::new().map_err(|e| LociError::TokenizerBuild {
            reason: format!("failed to create temp file for tokenizer config: {}", e),
        })?;
        write!(file, "{}", json_config).map_err(|e| LociError::TokenizerBuild {
            reason: format!("failed to write tokenizer config to temp file: {}", e),
        })?;
        Tokenizer::from_file(file.path())
            .map_err(|e| LociError::TokenizerBuild {
                reason: format!("failed to load tokenizer from json config: {}", e),
            })
    }

    fn tokenizer_from_config(&self, config: &TokenizerConfig) -> Result<Tokenizer, LociError> {
        match config.model_type.as_deref() {
            Some("gpt2") => self.build_bpe_tokenizer(config),
            Some(other) => Err(LociError::TokenizerBuild {
                reason: format!("unknown model type: {}", other),
            }),
            None => Err(LociError::TokenizerBuild {
                reason: "model_type is required to build tokenizer".into(),
            }),
        }
    }

    fn build_bpe_tokenizer(&self, config: &TokenizerConfig) -> Result<Tokenizer, LociError> {
        let tokens = config.tokens.as_ref().ok_or_else(|| {
            LociError::TokenizerBuild {
                reason: "tokens are required to build tokenizer but were not found".into(),
            }
        })?;
        let vocab: AHashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.to_owned(), i as u32))
            .collect();

        let merges_str = config.merges.as_ref().ok_or_else(|| {
            LociError::TokenizerBuild {
                reason: "merges are required to build tokenizer but were not found".into(),
            }
        })?;
        let merges: Vec<(String, String)> = merges_str
            .iter()
            .filter_map(|m| {
                let parts: Vec<&str> = m.split_whitespace().collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect();

        let model = BPE::builder()
            .byte_fallback(true)
            .vocab_and_merges(vocab, merges)
            .build()
            .map_err(|e| LociError::TokenizerBuild {
                reason: format!("failed to build BPE model: {}", e),
            })?;

        let mut tokenizer = Tokenizer::new(model);
        let pre_wrapper = self.configure_pre_tokenizer(&config.pre_tokenizer_tag)?;

        tokenizer.with_pre_tokenizer(Some(pre_wrapper));
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));

        let special_ids = self.configure_special_tokens(&tokenizer, config)?;

        let special_tokens = special_ids.iter()
            .map(|&id| {
                tokenizer.id_to_token(id).ok_or_else(|| {
                LociError::TokenizerBuild { 
                    reason: format!("Token with id: {}, annotated as a special is not present in tokenizer", id), 
                }
                })
                .map(|token| (token, id))
            })      
            .collect::<Result<Vec<(String, u32)>, LociError>>()?;

        if !special_tokens.is_empty() {
            let special_added_tokens = special_tokens.iter()
                .map(|(content, _id)| 
                    AddedToken::from(content, true).single_word(true)
                )
                .collect::<Vec<AddedToken>>();

            tokenizer.add_special_tokens(&special_added_tokens);
        }
        
        self.set_post_processor(
            &mut tokenizer,
            config,
            special_tokens,
        )?;

        Ok(tokenizer)
    }

    fn configure_pre_tokenizer(&self, pre_tag_opt: &Option<String>) -> Result<PreTokenizerWrapper, LociError> {
        let pt_wrapper = match pre_tag_opt.as_deref() {
            Some("glm4") => {
                // 1. Define the GLM-4 specific regex
                let glm4_pattern = r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[^\S\r\n]+|\s*[ \t\x0b\x0c\r\n]+"#;

                // 2. Build the Pre-Tokenizer Sequence
                let split = Split::new(
                    glm4_pattern, 
                    tokenizers::SplitDelimiterBehavior::Isolated, 
                    false
                ).map_err(|e| LociError::Tokenization { source: e })?;
                
                Sequence::new(vec![
                    split.into(),
                    ByteLevel::default().into(),
                ]).into()
            }
            _ => ByteLevel::default().into()
        };

        Ok(pt_wrapper)
    }

    fn configure_special_tokens(
        &self,
        tokenizer: &Tokenizer,
        config: &TokenizerConfig,
    ) -> Result<BTreeSet<u32>, LociError> {
        let mut special_ids = BTreeSet::new();

        if let Some(types) = &config.token_type {
            for (i, &token_type) in types.iter().enumerate() {
                if token_type == 4 {
                    special_ids.insert(i as u32);
                }
            }
        }

        let explicit_keys = [
            config.bos_token_id,
            config.eos_token_id,
            config.unknown_token_id,
            config.eot_token_id,
            config.padding_token_id,
            config.eom_token_id,
        ];

        for id in explicit_keys {
            if let Some(value) = id {
                special_ids.insert(value);
            }
        }

        Ok(special_ids)
    }

    fn set_post_processor(
        &self, 
        tokenizer: &mut Tokenizer,
        config: &TokenizerConfig,
        special_tokens: Vec<(String, u32)>
    ) -> Result<(), LociError> {
        let mut template = "$A:0".to_string();
        if config.add_bos {
            let (bos_token_str, id) = special_tokens.iter()
                .find(|(token, id)| *id == self.bos_token_id)
                .ok_or_else(|| {
                    LociError::TokenizerBuild { 
                        reason: format!("bos_token is not present in special_tokens"), 
                    }
                })?;
            template = format!("{}:0 {}", bos_token_str, template);
        }
        if config.add_eos {
            let (eos_token_str, id) = special_tokens.iter()
                .find(|(token, id)| *id == self.eos_token_id)
                .ok_or_else(|| {
                    LociError::TokenizerBuild { 
                        reason: format!("eos_token is not present in special_tokens"), 
                    }
                })?;
            template = format!("{} {}:0", template, eos_token_str);
        }

        let active_specials: Vec<(&str, u32)> = special_tokens.iter()
            .filter(|(name, _)| template.contains(name))
            .map(|(name, id)| (name.as_str(), *id))
            .collect();

        let processor = TemplateProcessing::builder()
            .try_single(template.as_str())
            .map_err(|e| LociError::TokenizerBuild {
                reason: format!("failed to build template: {}", e),
            })?
            .special_tokens(active_specials)
            .build()
            .map_err(|e| LociError::TokenizerBuild {
                reason: format!("failed to build post processor: {}", e),
            });

        tokenizer.with_post_processor(processor.ok());
        Ok(())
    }
}
