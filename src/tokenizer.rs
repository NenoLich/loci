use ahash::AHashMap;
use std::io::Write;

use crate::gguf_types::{GGUFTokenizerConfig, GgufKVMeta};

use tokenizers::Encoding;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{AddedToken, Tokenizer};

use tempfile::NamedTempFile;

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
    bos_token_id: u32,
    eos_token_id: u32,
}

impl TokenizerService {
    pub fn encode(&self, text: &str) -> anyhow::Result<Encoding> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(encoding)
    }

    pub fn decode(&self, tokens: &[u32]) -> anyhow::Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    pub fn set_bos_token_id(&mut self, bos_token_id: u32) {
        self.bos_token_id = bos_token_id;
    }

    pub fn set_eos_token_id(&mut self, eos_token_id: u32) {
        self.eos_token_id = eos_token_id;
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn process_token(&self, state: &mut StreamState, token: u32) -> anyhow::Result<Option<String>> {
        let prev_text = self.decode(&state.tokens[state.prev_index..state.read_index])?;
        
        state.tokens.push(token);
        let text = self.decode(&state.tokens[state.prev_index..])?;

        // Check if the new text is longer AND doesn't end with the replacement character
        if text.len() > prev_text.len() && !text.ends_with('\u{FFFD}') {
            let text = text.split_at(prev_text.len()).1.to_string();
            state.prev_index = state.read_index;
            state.read_index = state.tokens.len();
            anyhow::Ok(Some(text))
        } else {
            anyhow::Ok(None)
        }
    }

    pub fn decode_rest(&self, state: &mut StreamState) -> anyhow::Result<Option<String>> {
        let prev_text = self.decode(&state.tokens[state.prev_index..state.read_index])?;
        let text = self.decode(&state.tokens[state.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len()).1.to_string();
            anyhow::Ok(Some(text))
        } else {
            anyhow::Ok(None)
        }
    }
}

pub struct TokenizerServiceBuilder;

impl TokenizerServiceBuilder {
    pub fn from_gguf_metadata(metadata: &[GgufKVMeta]) -> anyhow::Result<TokenizerService> {
        let config = GGUFTokenizerConfig::from(metadata);

        if config.model_type.is_none() {
            anyhow::bail!("Failed to retrieve model_type, which is required to build tokenizer")
        }

        let tokenizer = match &config {
            c if c.json_config.is_some() => {
                Self::tokenizer_from_json_key(c.json_config.as_ref().unwrap())?
            }
            _ => Self::tokenizer_from_config(&config)?,
        };

        let bos_token_id = config.bos_token_id.unwrap_or(1);
        let eos_token_id = config.eos_token_id.unwrap_or(7);

        Ok(TokenizerService {
            tokenizer,
            bos_token_id,
            eos_token_id,
        })
    }

    fn tokenizer_from_json_key(json_config: &str) -> anyhow::Result<Tokenizer> {
        let mut file = NamedTempFile::new()?;
        write!(file, "{}", json_config)?;
        Tokenizer::from_file(file.path())
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from json config: {}", e))
    }

    fn tokenizer_from_config(config: &GGUFTokenizerConfig) -> anyhow::Result<Tokenizer> {
        match config.model_type.as_ref() {
            None => {
                anyhow::bail!("Failed to retrieve model_type, which is required to build tokenizer")
            }
            Some(s) if s == "gpt2" => Self::build_bpe_tokenizer(config),
            _ => anyhow::bail!(
                "Unknown model type: {}",
                config.model_type.as_ref().unwrap()
            ),
        }
    }

    fn build_bpe_tokenizer(config: &GGUFTokenizerConfig) -> anyhow::Result<Tokenizer> {
        let bos_token_id = config.bos_token_id;
        let eos_token_id = config.eos_token_id;

        let tokens = config.tokens.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Failed to retrieve tokens, which is required to build tokenizer")
        })?;
        let vocab: AHashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.to_owned(), i as u32))
            .collect();

        let merges_str = config.merges.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Failed to retrieve merges, which is required to build tokenizer")
        })?;
        let merges: Vec<(String, String)> = merges_str
            .into_iter()
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
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
        let (bos_token, eos_token) =
            Self::configure_special_tokens(&tokenizer, bos_token_id, eos_token_id)?;

        let special_tokens = vec![
            AddedToken::from(&bos_token, true).single_word(true),
            AddedToken::from(&eos_token, true).single_word(true),
        ];

        tokenizer.add_special_tokens(&special_tokens);

        Self::set_post_processor(
            &mut tokenizer,
            config.add_bos,
            config.add_eos,
            &bos_token,
            &eos_token,
            bos_token_id.unwrap_or(1),
            eos_token_id.unwrap_or(7),
        )?;

        Ok(tokenizer)
    }

    fn configure_special_tokens(
        tokenizer: &Tokenizer,
        bos_token_id: Option<u32>,
        eos_token_id: Option<u32>,
    ) -> anyhow::Result<(String, String)> {
        let bos_token = match bos_token_id {
            Some(id) => tokenizer.id_to_token(id).ok_or_else(|| {
                anyhow::anyhow!("bos_token_id from config is not present in tokenizer")
            })?,
            None => "<s>".to_string(),
        };

        let eos_token = match eos_token_id {
            Some(id) => tokenizer.id_to_token(id).ok_or_else(|| {
                anyhow::anyhow!("eos_token_id from config is not present in tokenizer")
            })?,
            None => "</s>".to_string(),
        };

        Ok((bos_token, eos_token))
    }

    fn set_post_processor(
        tokenizer: &mut Tokenizer,
        add_bos: bool,
        add_eos: bool,
        bos_token_str: &str,
        eos_token_str: &str,
        bos_token_id: u32,
        eos_token_id: u32,
    ) -> anyhow::Result<()> {
        let mut template = "$0:0".to_string();
        if add_bos {
            template = format!("{}:0 {}", bos_token_str, template);
        }
        if add_eos {
            template = format!("{} {}:0", template, eos_token_str);
        }

        let processor = TemplateProcessing::builder()
            .try_single(template.as_str())
            .map_err(|e| anyhow::anyhow!("Failed to build template: {}", e))?
            .special_tokens(vec![
                (bos_token_str, bos_token_id),
                (eos_token_str, eos_token_id),
            ])
            .build()
            .map_err(|e| anyhow::anyhow!(e));

        tokenizer.with_post_processor(processor.ok());
        Ok(())
    }  
}
