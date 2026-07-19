use ahash::AHashMap;
#[cfg(any(test, feature = "mock"))]
use mockall::automock;
use regex::Regex;
use serde_json::ser::Formatter;
use std::collections::BTreeSet;
use tokenizers::pre_tokenizers::sequence::Sequence;

use crate::config::TokenizerConfig;
use crate::error::LociError;
use crate::gguf::GgufInfo;
use crate::types::*;

use minijinja::{Environment, context};
use once_cell::sync::OnceCell;
use std::io;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::{PreTokenizerWrapper, byte_level::ByteLevel, split::Split};
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{AddedToken, Tokenizer as RawTokenizer};
use tracing::debug;

/// Minimal streaming context for incremental token-to-text decoding.
/// Accumulates tokens until a complete UTF-8 sequence is decodable, then clears.
#[derive(Debug)]
pub struct StreamContext {
    pub ids: Vec<u32>,
    pub prefix: String,
    pub prefix_index: usize,
}

impl StreamContext {
    /// Create a new streaming context with pre-allocated capacity.
    /// Typical token buffering is 1-3 tokens before UTF-8 boundary, so capacity of 8 is safe.
    pub fn with_capacity(capacity: usize) -> Self {
        StreamContext {
            ids: Vec::with_capacity(capacity),
            prefix: String::with_capacity(capacity),
            prefix_index: 0,
        }
    }

    /// Reset the streaming context by clearing all pending tokens.
    pub fn reset(&mut self) {
        self.ids.clear();
        self.prefix.clear();
        self.prefix_index = 0;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SpacedFormatter;

impl Formatter for SpacedFormatter {
    #[inline]
    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(b": ")
    }

    #[inline]
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if !first {
            writer.write_all(b", ")
        } else {
            Ok(())
        }
    }

    #[inline]
    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if !first {
            writer.write_all(b", ")
        } else {
            Ok(())
        }
    }
}

#[cfg_attr(any(test, feature = "mock"), automock)]
pub trait Tokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, LociError>;
    fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String, LociError>;
    fn eos_token_id(&self) -> u32;
    fn process_token_stream(
        &self,
        ctx: &mut StreamContext,
        token: u32,
    ) -> Result<Option<String>, LociError>;
    fn process_multiple_token_stream(
        &self,
        ctx: &mut StreamContext,
        tokens: &[u32],
    ) -> Result<Option<String>, LociError>;
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        raw_tools: &[Tool],
        enable_thinking: bool,
        flatten_tools_to_functions: bool,
    ) -> Result<String, LociError>;
    fn special_token_ids(&self) -> Vec<u32>;
}

pub struct TokenizerService {
    tokenizer: RawTokenizer,
    chat_template: String,
    bos_token_id: u32,
    eos_token_id: u32,
    eot_token_id: u32,
    eom_token_id: u32,
    unknown_token_id: u32,
    padding_token_id: u32,
    bos_token: OnceCell<String>,
    eos_token: OnceCell<String>,
    eot_token: OnceCell<String>,
    eom_token: OnceCell<String>,
    unknown_token: OnceCell<String>,
    padding_token: OnceCell<String>,
}

impl Tokenizer for TokenizerService {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, LociError> {
        self.tokenizer
            .encode(text, add_special_tokens)
            .map(|enc| enc.get_ids().to_vec())
            .map_err(|e| LociError::Tokenization { source: e })
    }

    fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String, LociError> {
        self.tokenizer
            .decode(tokens, skip_special_tokens)
            .map_err(|e| LociError::Tokenization { source: e })
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    fn special_token_ids(&self) -> Vec<u32> {
        self.tokenizer
            .get_added_vocabulary()
            .get_added_tokens_decoder()
            .iter()
            .filter(|&(_, token)| token.special)
            .map(|(id, _)| *id)
            .collect::<Vec<u32>>()
    }

    fn process_token_stream(
        &self,
        ctx: &mut StreamContext,
        token: u32,
    ) -> Result<Option<String>, LociError> {
        if ctx.prefix.is_empty() && !ctx.ids.is_empty() {
            let new_prefix = self.decode(ctx.ids.as_slice(), true)?;
            if !new_prefix.ends_with('�') {
                ctx.prefix = new_prefix;
                ctx.prefix_index = ctx.ids.len();
            }
        }

        ctx.ids.push(token);
        let text = self.decode(ctx.ids.as_slice(), true)?;
        if text.len() > ctx.prefix.len() && !text.ends_with('�') {
            if !(text.starts_with(&ctx.prefix)) {
                return Err(LociError::Tokenization {
                    source: "Invalid prefix in stream context".into(),
                });
            }

            let new_text = &text[ctx.prefix.len()..].to_string();
            let new_prefix_index = ctx.ids.len() - ctx.prefix_index;
            ctx.ids = ctx.ids.drain(ctx.prefix_index..).collect();
            ctx.prefix = self.decode(ctx.ids.as_slice(), true)?;
            ctx.prefix_index = new_prefix_index;
            Ok(Some(new_text.to_string()))
        } else {
            Ok(None)
        }
    }

    fn process_multiple_token_stream(
        &self,
        ctx: &mut StreamContext,
        tokens: &[u32],
    ) -> Result<Option<String>, LociError> {
        if ctx.prefix.is_empty() && !ctx.ids.is_empty() {
            let new_prefix = self.decode(ctx.ids.as_slice(), true)?;
            if !new_prefix.ends_with('�') {
                ctx.prefix = new_prefix;
                ctx.prefix_index = ctx.ids.len();
            }
        }

        ctx.ids.extend_from_slice(tokens);
        let text = self.decode(ctx.ids.as_slice(), true)?;
        if text.len() > ctx.prefix.len() && !text.ends_with('�') {
            if !(text.starts_with(&ctx.prefix)) {
                return Err(LociError::Tokenization {
                    source: "Invalid prefix in stream context".into(),
                });
            }

            let new_text = &text[ctx.prefix.len()..].to_string();
            let new_prefix_index = ctx.ids.len() - ctx.prefix_index;
            ctx.ids = ctx.ids.drain(ctx.prefix_index..).collect();
            ctx.prefix = self.decode(ctx.ids.as_slice(), true)?;
            ctx.prefix_index = new_prefix_index;
            Ok(Some(new_text.to_string()))
        } else {
            Ok(None)
        }
    }

    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        raw_tools: &[Tool],
        enable_thinking: bool,
        flatten_tools_to_functions: bool,
    ) -> Result<String, LociError> {
        let mut env = Environment::new();
        let name = "chat";
        env.add_template(name, &self.chat_template)
            .map_err(|e| LociError::Tokenization {
                source: Box::new(e),
            })?;
        let template = env
            .get_template(name)
            .map_err(|e| LociError::Tokenization {
                source: Box::new(e),
            })?;
        let bos_token = self
            .bos_token
            .get_or_try_init(|| self.decode(&[self.bos_token_id], false))?;
        let eos_token = self
            .eos_token
            .get_or_try_init(|| self.decode(&[self.eos_token_id], false))?;
        let eot_token = self
            .eot_token
            .get_or_try_init(|| self.decode(&[self.eot_token_id], false))?;
        let eom_token = self
            .eom_token
            .get_or_try_init(|| self.decode(&[self.eom_token_id], false))?;
        let unknown_token = self
            .unknown_token
            .get_or_try_init(|| self.decode(&[self.unknown_token_id], false))?;
        let padding_token = self
            .padding_token
            .get_or_try_init(|| self.decode(&[self.padding_token_id], false))?;

        // Build tools as list of JSON strings with proper field ordering
        let tools_json_list = if flatten_tools_to_functions {
            raw_tools
                .iter()
                .map(|tool| to_spaced_string(&tool.function))
                .collect::<Result<Vec<String>, serde_json::Error>>()
        } else {
            raw_tools
                .iter()
                .map(to_spaced_string)
                .collect::<Result<Vec<String>, serde_json::Error>>()
        }
        .map_err(|e| LociError::Tokenization {
            source: Box::new(e),
        })?;

        let rendered = template
            .render(context! {
                bos_token => bos_token,
                eos_token => eos_token,
                eot_token => eot_token,
                eom_token => eom_token,
                unknown_token => unknown_token,
                padding_token => padding_token,
                clear_thinking => false,
                messages => messages,
                add_generation_prompt => true,
                enable_thinking => enable_thinking,
                tools => tools_json_list,
            })
            .map_err(|e| LociError::Tokenization {
                source: Box::new(e),
            })?;

        Ok(rendered)
    }
}

impl Tokenizer for Box<dyn Tokenizer + Send + Sync + '_> {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, LociError> {
        (**self).encode(text, add_special_tokens)
    }

    fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String, LociError> {
        (**self).decode(tokens, skip_special_tokens)
    }

    fn eos_token_id(&self) -> u32 {
        (**self).eos_token_id()
    }

    fn process_token_stream(
        &self,
        ctx: &mut StreamContext,
        token: u32,
    ) -> Result<Option<String>, LociError> {
        (**self).process_token_stream(ctx, token)
    }

    fn process_multiple_token_stream(
        &self,
        ctx: &mut StreamContext,
        tokens: &[u32],
    ) -> Result<Option<String>, LociError> {
        (**self).process_multiple_token_stream(ctx, tokens)
    }

    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        raw_tools: &[Tool],
        enable_thinking: bool,
        flatten_tools_to_functions: bool,
    ) -> Result<String, LociError> {
        (**self).apply_chat_template(
            messages,
            raw_tools,
            enable_thinking,
            flatten_tools_to_functions,
        )
    }

    fn special_token_ids(&self) -> Vec<u32> {
        (**self).special_token_ids()
    }
}

impl TokenizerService {
    pub fn builder() -> TokenizerServiceBuilder {
        TokenizerServiceBuilder::default()
    }
}

fn to_spaced_string<T: serde::Serialize>(value: &T) -> Result<String, serde_json::Error> {
    let mut buf = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut buf, SpacedFormatter);
    value.serialize(&mut serializer)?;
    Ok(String::from_utf8(buf).unwrap())
}

pub struct TokenizerServiceBuilder {
    chat_template: String,
    bos_token_id: u32,
    eos_token_id: u32,
    eot_token_id: u32,
    eom_token_id: u32,
    unknown_token_id: u32,
    padding_token_id: u32,
    config: Option<TokenizerConfig>,
    python_get_pattern: Option<Regex>,
    tojson_kwarg_re: Option<Regex>,
}

impl Default for TokenizerServiceBuilder {
    fn default() -> Self {
        // Matches .get('key') or .get("key") including variations in spacing
        let python_get_pattern = Regex::new("\\.get\\(\\s*(['\"])([^'\"]+)['\"]\\s*\\)").ok();
        // Matches ensure_ascii kwarg in tojson() calls (e.g., tojson(x, ensure_ascii=False))
        let tojson_kwarg_re = Regex::new(r"(?:,\s*)?ensure_ascii\s*=\s*(?:True|False)").ok();

        Self {
            chat_template: "{% for message in messages %}{{ message.content }}{% endfor %}"
                .to_string(),
            bos_token_id: 1,
            eos_token_id: 2,
            eot_token_id: 2,
            eom_token_id: 2,
            unknown_token_id: 0,
            padding_token_id: 2,
            config: None,
            python_get_pattern,
            tojson_kwarg_re,
        }
    }
}

impl TokenizerServiceBuilder {
    pub fn with_gguf_metadata(mut self, info: &GgufInfo) -> Self {
        let metadata = info.kv_meta.as_slice();
        self.config = Some(TokenizerConfig::from(metadata));
        self
    }

    #[tracing::instrument(level = "debug", skip_all)]
    pub fn build(&mut self) -> Result<TokenizerService, LociError> {
        if self.config.is_none() {
            return Err(LociError::TokenizerBuild {
                reason: "TokenizerConfig is required to build tokenizer but was not set".into(),
            });
        }

        let config = self.config.as_ref().unwrap();
        if config.model_type.is_none() {
            return Err(LociError::TokenizerBuild {
                reason: "model_type is required to build tokenizer but was not found in config"
                    .into(),
            });
        }

        let cleaned_template = self.clean_chat_template(config.chat_template.as_deref());
        if let Some(ct) = cleaned_template {
            self.chat_template = ct;
        }
        if let Some(id) = config.bos_token_id {
            self.bos_token_id = id;
        }
        if let Some(id) = config.eos_token_id {
            self.eos_token_id = id;
        }
        if let Some(id) = config.eot_token_id {
            self.eot_token_id = id;
        }
        if let Some(id) = config.eom_token_id {
            self.eom_token_id = id;
        }
        if let Some(id) = config.unknown_token_id {
            self.unknown_token_id = id;
        }
        if let Some(id) = config.padding_token_id {
            self.padding_token_id = id;
        }

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
            unknown_token_id: self.unknown_token_id,
            padding_token_id: self.padding_token_id,
            bos_token: OnceCell::new(),
            eos_token: OnceCell::new(),
            eot_token: OnceCell::new(),
            eom_token: OnceCell::new(),
            unknown_token: OnceCell::new(),
            padding_token: OnceCell::new(),
        })
    }

    fn tokenizer_from_json_key(&self, json_config: &str) -> Result<RawTokenizer, LociError> {
        RawTokenizer::from_bytes(json_config.as_bytes()).map_err(|e| LociError::TokenizerBuild {
            reason: format!("failed to load tokenizer from json config string: {}", e),
        })
    }

    fn tokenizer_from_config(&self, config: &TokenizerConfig) -> Result<RawTokenizer, LociError> {
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

    fn build_bpe_tokenizer(&self, config: &TokenizerConfig) -> Result<RawTokenizer, LociError> {
        let tokens = config
            .tokens
            .as_ref()
            .ok_or_else(|| LociError::TokenizerBuild {
                reason: "tokens are required to build tokenizer but were not found".into(),
            })?;
        let vocab: AHashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.to_owned(), i as u32))
            .collect();

        let merges_str = config
            .merges
            .as_ref()
            .ok_or_else(|| LociError::TokenizerBuild {
                reason: "merges are required to build tokenizer but were not found".into(),
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

        let mut tokenizer = RawTokenizer::new(model);
        let pre_wrapper = self.configure_pre_tokenizer(&config.pre_tokenizer_tag)?;

        tokenizer.with_pre_tokenizer(Some(pre_wrapper));
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));

        let special_ids = self.configure_special_tokens(config)?;

        let special_tokens = special_ids
            .iter()
            .map(|&id| {
                tokenizer
                    .id_to_token(id)
                    .ok_or_else(|| LociError::TokenizerBuild {
                        reason: format!(
                            "Token with id: {}, annotated as a special is not present in tokenizer",
                            id
                        ),
                    })
                    .map(|token| (token, id))
            })
            .collect::<Result<Vec<(String, u32)>, LociError>>()?;

        if !special_tokens.is_empty() {
            let special_added_tokens = special_tokens
                .iter()
                .map(|(content, _id)| AddedToken::from(content, true).single_word(true))
                .collect::<Vec<AddedToken>>();

            tokenizer.add_special_tokens(&special_added_tokens);
        }

        let template_processing = self.set_post_processor(config, special_tokens)?;
        tokenizer.with_post_processor(template_processing);

        Ok(tokenizer)
    }

    fn configure_pre_tokenizer(
        &self,
        pre_tag_opt: &Option<String>,
    ) -> Result<PreTokenizerWrapper, LociError> {
        let pt_wrapper = match pre_tag_opt.as_deref() {
            Some("glm4") => {
                // 1. Define the GLM-4 specific regex
                let glm4_pattern = r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[^\S\r\n]+|\s*[ \t\x0b\x0c\r\n]+"#;

                // 2. Build the Pre-Tokenizer Sequence
                let split = Split::new(
                    glm4_pattern,
                    tokenizers::SplitDelimiterBehavior::Isolated,
                    false,
                )
                .map_err(|e| LociError::Tokenization { source: e })?;

                Sequence::new(vec![split.into(), ByteLevel::default().into()]).into()
            }
            _ => ByteLevel::default().into(),
        };

        Ok(pt_wrapper)
    }

    fn configure_special_tokens(
        &self,
        config: &TokenizerConfig,
    ) -> Result<BTreeSet<u32>, LociError> {
        let mut special_ids = BTreeSet::new();

        if let Some(types) = &config.token_type {
            for (i, &token_type) in types.iter().enumerate() {
                if token_type == 4 || token_type == 3 {
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

        for value in explicit_keys.iter().flatten() {
            special_ids.insert(*value);
        }

        Ok(special_ids)
    }

    fn set_post_processor(
        &self,
        config: &TokenizerConfig,
        special_tokens: Vec<(String, u32)>,
    ) -> Result<Option<TemplateProcessing>, LociError> {
        let mut template = "$A:0".to_string();
        if config.add_bos {
            let (bos_token_str, _) = special_tokens
                .iter()
                .find(|(_, id)| *id == self.bos_token_id)
                .ok_or_else(|| LociError::TokenizerBuild {
                    reason: "bos_token is not present in special_tokens".to_string(),
                })?;
            template = format!("{}:0 {}", bos_token_str, template);
        }
        if config.add_eos {
            let (eos_token_str, _) = special_tokens
                .iter()
                .find(|(_, id)| *id == self.eos_token_id)
                .ok_or_else(|| LociError::TokenizerBuild {
                    reason: "eos_token is not present in special_tokens".to_string(),
                })?;
            template = format!("{} {}:0", template, eos_token_str);
        }

        let active_specials: Vec<(&str, u32)> = special_tokens
            .iter()
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

        Ok(processor.ok())
    }

    fn clean_chat_template(&self, template: Option<&str>) -> Option<String> {
        let template = template?;
        let mut env = Environment::new();
        let name = "chat_check";

        // Step 1: Always try to compile and immediately test-render it
        if let (Ok(()), Ok(tmpl)) = (env.add_template(name, template), env.get_template(name)) {
            // We must mock a render to see if it actually executes without errors
            let mock_messages = [
                ChatMessage {
                    role: Role::System,
                    content: Some("You are a helpful assistant".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
                ChatMessage {
                    role: Role::User,
                    content: Some("Hello".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
            ];
            let mock_tools = [
                Tool {
                    r#type: "Tool 1".to_string(),
                    function: Function {
                        name: "tool_1".to_string(),
                        description: Some("This is tool 1".to_string()),
                        parameters: FunctionParameters {
                            r#type: "object".to_string(),
                            properties: None,
                            required: vec![],
                        },
                    },
                },
                Tool {
                    r#type: "Tool 2".to_string(),
                    function: Function {
                        name: "tool_2".to_string(),
                        description: Some("This is tool 2".to_string()),
                        parameters: FunctionParameters {
                            r#type: "object".to_string(),
                            properties: None,
                            required: vec![],
                        },
                    },
                },
            ];
            let test_render = tmpl.render(context! {
                messages => mock_messages,
                tools => mock_tools,
            });

            match test_render {
                Ok(_) => return Some(template.to_string()),
                Err(err) => {
                    // Step 2: Dispatch on the error.
                    // MiniJinja throws a TemplateNotFound or UnknownMethodErrorKind dynamically.
                    let err_msg = err.to_string();
                    if err_msg.contains("no method named get")
                        || err.kind() == minijinja::ErrorKind::UnknownMethod
                    {
                        return self.fix_python_code(template);
                    } else if err_msg.contains("unknown keyword argument") {
                        return self.fix_tojson_kwargs(template);
                    } else {
                        debug!("Template validation failed with error: {}", err_msg);
                        return None;
                    }
                }
            }
        }

        None
    }

    fn fix_python_code(&self, template: &str) -> Option<String> {
        // Converts .get('tool_calls') into ['tool_calls']
        Some(if let Some(ref pattern) = self.python_get_pattern {
            pattern
                .replace_all(template, |caps: &regex::Captures<'_>| {
                    format!("[{}{}{}]", &caps[1], &caps[2], &caps[1])
                })
                .into_owned()
        } else {
            template.to_string()
        })
    }

    fn fix_tojson_kwargs(&self, template: &str) -> Option<String> {
        // Removes unsupported keyword arguments like ensure_ascii from tojson() calls
        if let Some(ref pattern) = self.tojson_kwarg_re {
            let result = pattern.replace_all(template, "").into_owned();
            // Clean up any artifacts from kwarg removal
            let result = result.replace(", ,", ",");
            let result = result.replace(",,", ",");
            let result = result.replace("(, ", "(");
            let result = result.replace("( ", "(");
            Some(result.replace(", )", ")"))
        } else {
            Some(template.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(12)]
    #[case(0)]
    fn test_stream_context_with_capacity(#[case] capacity: usize) {
        let ctx = StreamContext::with_capacity(capacity);
        assert!(ctx.ids.is_empty());
        assert_eq!(ctx.prefix, "");
        assert_eq!(ctx.prefix_index, 0);
    }

    #[test]
    fn test_stream_context_reset() {
        let mut ctx = StreamContext::with_capacity(12);
        ctx.ids = vec![1, 2, 3];
        ctx.prefix = "hello".to_string();
        ctx.prefix_index = 1;
        ctx.reset();
        assert!(ctx.ids.is_empty());
        assert_eq!(ctx.prefix, "");
        assert_eq!(ctx.prefix_index, 0);
    }

    // --- clean_chat_template ---

    #[rstest]
    #[case(None, None)]
    #[case(Some("".to_string()), Some("".to_string()))]
    #[case(Some("{{ messages }}".to_string()), Some("{{ messages }}".to_string()))]
    #[case(Some("{% if %}".to_string()), None)]
    fn test_clean_chat_template(
        #[case] template: Option<String>,
        #[case] expected: Option<String>,
    ) {
        let builder = TokenizerServiceBuilder::default();
        let result = builder.clean_chat_template(template.as_deref());
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case("{{ messages.get('tool_calls') }}", "{{ messages['tool_calls'] }}")]
    #[case("{{ messages.get(\"tool_calls\") }}", "{{ messages[\"tool_calls\"] }}")]
    #[case("{{ messages.get('a') }}", "{{ messages['a'] }}")]
    #[case("no get here", "no get here")]
    fn test_clean_chat_template_fix_python_code(#[case] template: &str, #[case] expected: &str) {
        let builder = TokenizerServiceBuilder::default();
        let result = builder.clean_chat_template(Some(template));
        assert_eq!(result, Some(expected.to_string()));
    }

    #[test]
    fn test_clean_chat_template_fix_tojson_kwargs() {
        let builder = TokenizerServiceBuilder::default();
        let result =
            builder.clean_chat_template(Some("{{ messages | tojson(ensure_ascii=False) }}"));
        assert_eq!(result, Some("{{ messages | tojson() }}".to_string()));
    }

    // --- fix_python_code (direct) ---

    #[rstest]
    #[case("{{ messages.get('') }}", "{{ messages.get('') }}")]
    #[case("{{ messages.get('tool_calls') }}", "{{ messages['tool_calls'] }}")]
    #[case(
        "{{ messages.get('tool_calls'), messages.get('tools') }}",
        "{{ messages['tool_calls'], messages['tools'] }}"
    )]
    #[case("{{ messages.get(\"tool_calls\") }}", "{{ messages[\"tool_calls\"] }}")]
    #[case("{{ messages.get( 'key' ) }}", "{{ messages['key'] }}")]
    #[case("no get", "no get")]
    #[case("", "")]
    fn test_fix_python_code(#[case] template: &str, #[case] expected: &str) {
        let builder = TokenizerServiceBuilder::default();
        assert_eq!(
            builder.fix_python_code(template),
            Some(expected.to_string())
        );
    }

    // --- fix_tojson_kwargs (direct) ---

    #[rstest]
    #[case("{{ x | tojson(ensure_ascii=False) }}", "{{ x | tojson() }}")]
    #[case("{{ x | tojson(ensure_ascii=True) }}", "{{ x | tojson() }}")]
    #[case(
        "{{ x | tojson(ensure_ascii=True, indent=2) }}",
        "{{ x | tojson(indent=2) }}"
    )]
    #[case(
        "{{ x | tojson(indent=2, ensure_ascii=False) }}",
        "{{ x | tojson(indent=2) }}"
    )]
    #[case(
        "{{ x | tojson(indent=2, ensure_ascii=False, sort_keys=True) }}",
        "{{ x | tojson(indent=2, sort_keys=True) }}"
    )]
    #[case("no tojson", "no tojson")]
    #[case("", "")]
    fn test_fix_tojson_kwargs(#[case] template: &str, #[case] expected: &str) {
        let builder = TokenizerServiceBuilder::default();
        assert_eq!(
            builder.fix_tojson_kwargs(template),
            expected.to_string().into()
        );
    }

    #[rstest]
    #[case(serde_json::to_value(Function {
        name: "foo".to_string(),
        description: Some("bar".to_string()),
        parameters: FunctionParameters {
            r#type: "object".to_string(),
            properties: Some(
                [].into_iter().collect(),
            ),
            required: vec![],
        },
        }).expect("to_value"),
        r#"{"name": "foo", "description": "bar", "parameters": {"type": "object", "properties": {}, "required": []}}"#
    )]
    #[case(serde_json::to_value(Function {
        name: "foo".to_string(),
        description: Some("bar".to_string()),
        parameters: FunctionParameters {
            r#type: "object".to_string(),
            properties: Some(
                [("key".to_string(), serde_json::json!(1))].into_iter().collect(),
            ),
            required: vec![],
        },
        }).expect("to_value"),
        r#"{"name": "foo", "description": "bar", "parameters": {"type": "object", "properties": {"key": 1}, "required": []}}"#
    )]
    #[case(serde_json::to_value(Function {
        name: "foo".to_string(),
        description: None,
        parameters: FunctionParameters {
            r#type: "object".to_string(),
            properties: Some(
                [
                    ("key1".to_string(), serde_json::json!(1)),
                    ("key2".to_string(), serde_json::json!(2)),
                ].into_iter().collect(),
            ),
            required: vec!["key1".to_string()],
        },
        }).expect("to_value"),
        r#"{"name": "foo", "description": null, "parameters": {"type": "object", "properties": {"key1": 1, "key2": 2}, "required": ["key1"]}}"#
    )]
    #[case(serde_json::to_value(Function {
        name: "foo".to_string(),
        description: Some("bar".to_string()),
        parameters: FunctionParameters {
            r#type: "object".to_string(),
            properties: Some(
                [("key".to_string(), serde_json::json!({"inner1": 1, "inner2": 2}))].into_iter().collect(),
            ),
            required: vec![],
        },
        }).expect("to_value"),
        r#"{"name": "foo", "description": "bar", "parameters": {"type": "object", "properties": {"key": {"inner1": 1, "inner2": 2}}, "required": []}}"#
    )]
    #[case(serde_json::to_value(Function {
        name: "foo".to_string(),
        description: Some("bar".to_string()),
        parameters: FunctionParameters {
            r#type: "object".to_string(),
            properties: Some(
                [("key".to_string(), serde_json::json!([1, 2, 3]))].into_iter().collect(),
            ),
            required: vec![],
        },
        }).expect("to_value"),
        r#"{"name": "foo", "description": "bar", "parameters": {"type": "object", "properties": {"key": [1, 2, 3]}, "required": []}}"#
    )]
    fn test_to_spaced_string(#[case] input_struct: serde_json::Value, #[case] expected: &str) {
        let spaced_string = to_spaced_string(&input_struct).expect("to_spaced_string");
        assert_eq!(spaced_string, expected);
    }

    #[rstest]
    #[case(
        TokenizerConfig {
            tokens: Some(vec![
                "bos".to_string(),
                "eos".to_string(),
                "pad".to_string(),
                "regular1".to_string(),
                "regular2".to_string(),
            ]),
            token_type: Some(vec![3, 4, 4, 1, 1]),
            ..Default::default()
        },
        BTreeSet::from([0, 1, 2])
    )]
    #[case(
        TokenizerConfig {
            tokens: Some(vec![
                "bos".to_string(),
                "eos".to_string(),
                "pad".to_string(),
                "regular1".to_string(),
                "regular2".to_string(),
                "eom".to_string(),
            ]),
            token_type: Some(vec![3, 4, 4, 1, 1, 1]),
            eom_token_id: Some(5),
            ..Default::default()
        },
        BTreeSet::from([0, 1, 2, 5])
    )]
    #[case(
        TokenizerConfig {
            tokens: Some(vec![
                "bos".to_string(),
                "eos".to_string(),
                "pad".to_string(),
                "regular1".to_string(),
                "regular2".to_string(),
                "eom".to_string(),
            ]),
            token_type: Some(vec![3, 4, 4, 1, 1, 1]),
            eom_token_id: Some(5),
            padding_token_id: Some(2),
            ..Default::default()
        },
        BTreeSet::from([0, 1, 2, 5])
    )]
    fn test_configure_special_tokens(
        #[case] config: TokenizerConfig,
        #[case] expected: BTreeSet<u32>,
    ) {
        let builder = TokenizerServiceBuilder::default();
        let special_tokens = builder
            .configure_special_tokens(&config)
            .expect("configure_special_tokens");
        assert_eq!(special_tokens, expected);
    }

    #[rstest]
    #[case(Some("glm4".to_string()))]
    #[case(None)]
    fn test_configure_pre_tokenizer_types(#[case] pre_tag_opt: Option<String>) {
        let builder = TokenizerServiceBuilder::default();
        let pt_wrapper = builder
            .configure_pre_tokenizer(&pre_tag_opt)
            .expect("Should configure successfully");
        match pre_tag_opt.as_deref() {
            Some("glm4") => {
                // Verify that "glm4" produces a Sequence wrapper variant
                assert!(
                    matches!(pt_wrapper, PreTokenizerWrapper::Sequence(_)),
                    "Expected PreTokenizerWrapper::Sequence for glm4, got {:?}",
                    pt_wrapper
                );
            }
            _ => {
                // Verify that default produces a pure ByteLevel wrapper variant
                assert!(
                    matches!(pt_wrapper, PreTokenizerWrapper::ByteLevel(_)),
                    "Expected PreTokenizerWrapper::ByteLevel for fallback, got {:?}",
                    pt_wrapper
                );
            }
        }
    }

    #[rstest]
    #[case(
        TokenizerConfig {
            token_type: Some(vec![1, 1, 1]),
            merges: Some(vec!["a b".to_string(), "b c".to_string()]),
            ..Default::default()
        },
        "tokens are required to build tokenizer but were not found"
    )]
    #[case(
        TokenizerConfig {
            tokens: Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
            token_type: Some(vec![1, 2, 3]),
            ..Default::default()
        },
        "merges are required to build tokenizer but were not found"
    )]
    fn test_build_bpe_tokenizer_failure(
        #[case] tokenizer_config: TokenizerConfig,
        #[case] expected_error_str: &str,
    ) {
        let builder = TokenizerServiceBuilder::default();
        let result = builder
            .build_bpe_tokenizer(&tokenizer_config)
            .expect_err("build_bpe_tokenizer should fail, but did not");
        assert!(result.to_string().contains(expected_error_str));
    }

    #[test]
    fn test_build_bpe_tokenizer_filter_merges() {
        let builder = TokenizerServiceBuilder::default();
        let tokenizer_config = TokenizerConfig {
            tokens: Some(vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "ab".to_string(),
                "bc".to_string(),
            ]),
            token_type: Some(vec![1, 2, 3]),
            merges: Some(vec![
                "a b".to_string(),
                "b c".to_string(),
                "a b c".to_string(),
            ]),
            ..Default::default()
        };
        let tokenizer = builder
            .build_bpe_tokenizer(&tokenizer_config)
            .expect("build_bpe_tokenizer should succeed");

        let model = tokenizer.get_model();
        let model_json = serde_json::to_value(model).expect("model serialize should succeed");
        let actual_merges = model_json
            .get("merges")
            .and_then(|m| m.as_array())
            .expect("BPE serialization should contain a merges array");

        assert!(
            actual_merges.contains(&serde_json::json!(["a", "b"])),
            "Expected merges to contain ['a', 'b'], but got: {:?}",
            actual_merges
        );
        assert!(
            actual_merges.contains(&serde_json::json!(["b", "c"])),
            "Expected merges to contain ['b', 'c'], but got: {:?}",
            actual_merges
        );
        assert!(!actual_merges.contains(&serde_json::json!("a b c")));
        assert!(!actual_merges.contains(&serde_json::json!(["a", "b", "c"])));
    }

    #[rstest]
    #[case(
        TokenizerConfig {
            tokens: Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
            token_type: Some(vec![1, 1, 1]),
            merges: Some(vec!["a b".to_string(), "b c".to_string()]),
            add_bos: true,
            ..Default::default()
        },
        vec![(String::from("eos_token"), 2), (String::from("unknown_token"), 0)],
        "bos_token is not present in special_tokens"
    )]
    #[case(
        TokenizerConfig {
            tokens: Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
            token_type: Some(vec![1, 1, 1]),
            merges: Some(vec!["a b".to_string(), "b c".to_string()]),
            add_eos: true,
            ..Default::default()
        },
        vec![(String::from("bos_token"), 1), (String::from("unknown_token"), 0)],
        "eos_token is not present in special_tokens"
    )]
    fn test_set_postprocessor_failure(
        #[case] config: TokenizerConfig,
        #[case] special_tokens: Vec<(String, u32)>,
        #[case] expected_error_str: &str,
    ) {
        let builder = TokenizerServiceBuilder::default();
        let result = builder
            .set_post_processor(&config, special_tokens)
            .expect_err("set_postprocessor should fail, but did not");
        assert!(result.to_string().contains(expected_error_str));
    }

    #[rstest]
    #[case(
        TokenizerConfig {
            tokens: Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
            token_type: Some(vec![1, 1, 1]),
            merges: Some(vec!["a b".to_string(), "b c".to_string()]),
            ..Default::default()
        },
        vec![],
        ""
    )]
    #[case(
        TokenizerConfig {
            tokens: Some(vec!["a".to_string(), "bos_token".to_string(), "c".to_string()]),
            token_type: Some(vec![1, 1, 1]),
            merges: Some(vec!["a b".to_string(), "b c".to_string()]),
            add_bos: true,
            bos_token_id: Some(1),
            ..Default::default()
        },
        vec![(String::from("bos_token"), 1), (String::from("pad_token"), 2)],
        "bos_token"
    )]
    #[case(
        TokenizerConfig {
            tokens: Some(vec!["a".to_string(), "b".to_string(), "eos_token".to_string()]),
            token_type: Some(vec![1, 1, 1]),
            merges: Some(vec!["a b".to_string(), "b c".to_string()]),
            add_eos: true,
            eos_token_id: Some(2),
            ..Default::default()
        },
        vec![(String::from("bos_token"), 1), (String::from("eos_token"), 2)],
        "eos_token"
    )]
    fn test_set_postprocessor_success(
        #[case] config: TokenizerConfig,
        #[case] special_tokens: Vec<(String, u32)>,
        #[case] expected_token_in_template: &str,
    ) {
        let builder = TokenizerServiceBuilder::default();
        let result = builder
            .set_post_processor(&config, special_tokens)
            .expect("set_postprocessor should succeed, but did not");
        assert!(result.is_some());
        let template_processing = result.unwrap();
        let template_json_str = serde_json::to_string(&template_processing)
            .expect("template_processing serialize should succeed");
        assert!(template_json_str.contains(expected_token_in_template));
    }
}
