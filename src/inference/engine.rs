use std::path::{Path, PathBuf};
use std::time::Instant;
use std::rc::Rc;
use std::mem::take;

use crate::error::LociError;
use crate::gguf::{GgufInfo, Loader};
use crate::model::{MixedCache, Model, ModelBuilder};
use crate::config::{ModelConfig, GenerationConfig, InferenceConfig, GenerationOverrides, GenerationConfigBuilder, ModelCacheConfig};
use crate::tokenizer::{StreamContext, TokenizerService, TokenizerServiceBuilder};
use crate::api::types::{Role, ChatMessage, Tool, FinishReason, ReasoningEffort, Usage, CompletionTokensDetails, LogprobsContent, TopLogprobs, ChunkToolCall, ToolCall};
use crate::inference::{DeviceManager, InferenceSampler, ReasoningSupervisor, ToolCallingSupervisor, StopPatternMatcher, SamplingResult, ToolFormatStyle, GenerationDataType, GenerationEvent, GenerationReport, StreamCallback, PostSamplingConfig, StreamFrame, GenerationHandler, ModelCacheManager, MatchCacheResult, GenerationContext};
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::quantized_var_builder::VarBuilder;
use tokenizers::tokenizer::Encoding;
use memmap2::MmapOptions;
use once_cell::sync::OnceCell;
use tracing::{debug, warn};
use nvtx::{range_push, range_pop};

pub struct InferenceEngineBuilder {
    gguf_path: Option<PathBuf>,
    inference_config: Option<InferenceConfig>,
    model_cache_config: Option<ModelCacheConfig>
}

impl InferenceEngineBuilder {
    pub fn new() -> Self {
        Self {
            gguf_path: None,
            inference_config: None,
            model_cache_config: None
        }
    }

    pub fn with_gguf_metadata(mut self, path: impl AsRef<Path>) -> Self {
        self.gguf_path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn inference_config(mut self, inference_config: Option<InferenceConfig>) -> Self {
        self.inference_config = inference_config;
        self
    }

    pub fn model_cache_config(mut self, model_cache_config: Option<ModelCacheConfig>) -> Self {
        self.model_cache_config = model_cache_config;
        self
    }

    fn init_model(&self, model_config: &ModelConfig, inference_config: &InferenceConfig, device: &Device) -> Result<Box<dyn Model + Send + Sync>, LociError> {
        let start_time = Instant::now();
        range_push!("VarBuilder Init");
        debug!("Creating VarBuilder...");
        let file = std::fs::File::open(model_config.file_path.clone())?;
        let mmap = unsafe {
            MmapOptions::new().map(&file)
                .map_err(|e| LociError::ModelLoad(e.to_string()))?
        };
        let var_builder = VarBuilder::from_gguf_buffer(&mmap, device)
            .map_err(|e| LociError::ModelLoad(e.to_string()))?;

        debug!("VarBuilder created");
        range_pop!();

        let model = ModelBuilder::new(model_config.clone(), var_builder, inference_config).build()?;
        debug!("Model loaded in {:.3}s", start_time.elapsed().as_secs_f32());
        Ok(model)
    }

    pub fn build(self) -> Result<InferenceEngine, LociError> {
        let gguf_path = self.gguf_path.as_deref().ok_or_else(|| {
            LociError::ModelLoad("gguf_path is required but was not set".into())
        })?;

        let gguf_info = Loader::load_gguf_info(gguf_path, 0, false)?;

        let gen_builder = GenerationConfig::builder()
            .with_gguf_metadata(&gguf_info)
            .map_err(|e| LociError::ModelLoad(e.to_string()))?;
 
        range_push!("Tokenizer build");
        let tokenizer = TokenizerService::builder()
            .with_gguf_metadata(&gguf_info)
            .build()?;
        range_pop!();
        let device = DeviceManager::select()?;

        let model_config = ModelConfig::from_gguf_info(&gguf_info).map_err(|e| {
            LociError::ModelLoad(format!("failed to parse model config: {}", e))
        })?;

        let model_name = model_config.model_name.clone();
        let vocab_size = model_config.vocab_size;
        let supports_reasoning = model_config.supports_reasoning;
        let supports_tool_calling = model_config.supports_tool_calling;
        let post_sampling_config = PostSamplingConfig {
            tool_call_start_token_id: model_config.tool_call_start_token_id,
            tool_call_end_token_id: model_config.tool_call_end_token_id,
            reasoning_start_token_id: model_config.reasoning_start_token_id,
            reasoning_end_token_id: model_config.reasoning_end_token_id,
            tool_call_format_style: model_config.tool_call_format_style.clone(),
            arg_key_open_token_id: model_config.arg_key_open_token_id,
            arg_key_close_token_id: model_config.arg_key_close_token_id,
            arg_value_open_token_id: model_config.arg_value_open_token_id,
            arg_value_close_token_id: model_config.arg_value_close_token_id,
        };
        let inference_config = match self.inference_config {
            Some(ref config) => &config,
            None => &InferenceConfig::default(),
        };
        let flash_attn = inference_config.flash_attn;
        let prefix_caching = inference_config.prefix_caching;
        let generation_context = GenerationContext::new(&model_name);

        let model = self.init_model(&model_config, inference_config, &device)?;

        let model_cache_config = match self.model_cache_config {
            Some(config) => config,
            None => ModelCacheConfig::default(),
        };

        let model_cache_manager = ModelCacheManager::builder(&model_name)
            .with_config(model_cache_config)
            .prefix_caching(prefix_caching)
            .build()
            .map_err(|e| LociError::Cache(e.to_string()))?;

        Ok(InferenceEngine {
            tokenizer,
            model_cache_manager,
            generation_context,
            device,
            model_path: gguf_path.to_string_lossy().into(),
            vocab_size,
            model,
            flash_attn,
            prefix_caching,
            supports_reasoning,
            supports_tool_calling,
            flatten_tools_to_functions: model_config.flatten_tools_to_functions,
            post_sampling_config,
            gen_builder,
        })
    }
}

pub struct InferenceEngine {
    tokenizer: TokenizerService,
    model_cache_manager: ModelCacheManager,
    generation_context: GenerationContext,
    device: Device,
    model_path: String,
    vocab_size: usize,
    model: Box<dyn Model + Send + Sync>,
    flash_attn: bool,
    prefix_caching: bool,
    supports_reasoning: bool,
    supports_tool_calling: bool,
    flatten_tools_to_functions: bool,
    post_sampling_config: PostSamplingConfig,
    gen_builder: GenerationConfigBuilder,
}

impl InferenceEngine {
    pub fn builder() -> InferenceEngineBuilder {
        InferenceEngineBuilder::new()
    }

    pub fn model_path(&self) -> String {
        self.model_path.clone()
    }

    pub fn generate_chat_stream(
        &mut self,
        messages: &[ChatMessage],
        tools: &[Tool],
        overrides: GenerationOverrides,
        callback: StreamCallback,
    ) -> Result<GenerationReport, LociError> 
    {
        let gen_config = self.gen_builder.clone().with_overrides(overrides).build();
        debug!("Generation parameters: {:#?}", gen_config);
        debug!("Using flash attention: {}", self.flash_attn);
        let enable_thinking = self.supports_reasoning && gen_config.reasoning_effort != ReasoningEffort::None;
        let prompt = self.tokenizer.apply_chat_template(messages, tools, enable_thinking, self.flatten_tools_to_functions)?;
        debug!("Model prompt: {:#?}", prompt); 
        let encoding = self.tokenizer.encode(&prompt, false)?;
        self.generate_from_encoding(encoding, gen_config, callback)
    }

    pub fn generate_stream(
        &mut self,
        prompt: &str,
        overrides: GenerationOverrides,
        callback: StreamCallback,
    ) -> Result<GenerationReport, LociError> 
    {
        let gen_config = self.gen_builder.clone().with_overrides(overrides).build();
        let encoding = self.tokenizer.encode(&prompt, true)?;
        self.generate_from_encoding(encoding, gen_config, callback)
    }

    pub fn generate_from_encoding(
        &mut self,
        encoding: Encoding,
        gen_config: GenerationConfig,
        mut callback: StreamCallback,
    ) -> Result<GenerationReport, LociError> 
    {
        // Tokenize prompt
        let prompt_token_ids = encoding.get_ids();
        let input_tokens_len = prompt_token_ids.len();
        debug!("Input tokens length: {}", input_tokens_len);

        // Initialize sampler (handles temperature, top-p, etc.)
        let tool_start_token_id = self.post_sampling_config.tool_call_start_token_id;
        let mut sampler = InferenceSampler::new(gen_config.clone(), self.vocab_size, 20, tool_start_token_id);
        prompt_token_ids.iter().for_each(|token| sampler.add_token(*token));

        let matched_cache_len = self.match_cache_ctx(prompt_token_ids)?;
        let input_tokens = &prompt_token_ids[matched_cache_len..];
        debug!(input_len = input_tokens.len());
        let end_token = self.tokenizer.eos_token_id();
        let mut stream_ctx = crate::tokenizer::StreamContext::with_capacity(8);

        let generation_start = Instant::now();
        let mut reasoning_supervisor = ReasoningSupervisor::new(self.supports_reasoning, &gen_config.reasoning_effort, &self.post_sampling_config);
        let mut tool_calling_supervisor = ToolCallingSupervisor::new(self.supports_tool_calling, &self.post_sampling_config, &self.tokenizer, &gen_config.tool_choice)
            .map_err(|e| LociError::Inference{ source: e.into_boxed_dyn_error() })?;
        let reasoning_budget = reasoning_supervisor.as_ref().map_or(0, |rs| rs.reasoning_budget as usize);

        let mut handler = GenerationHandler::new(
            input_tokens, 
            sampler, 
            reasoning_supervisor, 
            tool_calling_supervisor,
            end_token,
        );

        let mut stop_pattern_matcher = StopPatternMatcher::new(gen_config.stop_tokens, &self.tokenizer);

        let mut content_text = String::with_capacity(gen_config.max_tokens);
        let mut reasoning_text = String::with_capacity(reasoning_budget);
        let mut finish_reason = FinishReason::Stop;
        let mut i = 0;

        // Autoregressive generation loop
        'generation:loop {
            // This handles pre-fill on i=0, and single token generation on i>1
            let input_tokens = take(&mut handler.input_tokens);
            let input = Tensor::new(input_tokens.as_slice(), &self.device)
                .map_err(|e| LociError::Inference{ source: Box::new(e) })?
                .unsqueeze(0)
                .map_err(|e| LociError::Inference{ source: Box::new(e) })?;
            let pos = self.generation_context.token_ids.len();
            let logits = self.model.forward(&input, &mut self.generation_context.cache, pos, self.flash_attn)
                .map_err(|e| LociError::Inference{ source: e.into_boxed_dyn_error() })?;
            let squeezed_logits = self.squeeze_logits(logits)
                .map_err(|e| LociError::Inference{ source: e.into_boxed_dyn_error() })?;
            
            self.generation_context.token_ids.extend(input_tokens);
            handler.advance(&squeezed_logits, gen_config.logprobs, gen_config.top_logprobs)
                .map_err(|e| LociError::Inference{ source: e.into_boxed_dyn_error() })?;
            i += 1;

            for event in handler.take_pending_events() {
                match event {
                    GenerationEvent::GenerationStopped => {
                        break 'generation;
                    }
                    GenerationEvent::ContentSampled { sampling_result } => {
                        if stop_pattern_matcher.matches(sampling_result.token) {
                            finish_reason = FinishReason::Stop;
                            handler.soft_stop();
                        }
                        if let Some(output) = self.tokenizer.process_token_stream(&mut stream_ctx, sampling_result.token)? {
                            let logprobs = self.decode_sampling_result(&output, sampling_result);
                        
                            callback(StreamFrame {
                                output: &output,
                                tool_call_chunk: None,
                                output_type: GenerationDataType::DirectContent,
                                logprobs,
                            })?;
                            content_text.push_str(&output);
                        }
                    }
                    GenerationEvent::ReasoningSampled { sampling_result } => {
                        if stop_pattern_matcher.matches(sampling_result.token) {
                            finish_reason = FinishReason::Stop;
                            handler.soft_stop();
                        }
                        if let Some(output) = self.tokenizer.process_token_stream(&mut stream_ctx, sampling_result.token)? {
                            let logprobs = self.decode_sampling_result(&output, sampling_result);
                        
                            callback(StreamFrame {
                                output: &output,
                                tool_call_chunk: None,
                                output_type: GenerationDataType::Reasoning,
                                logprobs,
                            })?;
                            reasoning_text.push_str(&output);
                        }    
                    }
                    GenerationEvent::ToolCallNameChunk { chunk } => {
                        callback(StreamFrame {
                            output: "",
                            tool_call_chunk: Some(chunk),
                            output_type: GenerationDataType::ToolCallName,
                            logprobs: None,
                        })?;
                        finish_reason = FinishReason::ToolCalls;
                    }
                    GenerationEvent::ToolCallArgumentsChunk { chunk } => {
                        callback(StreamFrame {
                            output: "",
                            tool_call_chunk: Some(chunk),
                            output_type: GenerationDataType::ToolCallArguments,
                            logprobs: None,
                        })?;
                    }
                    _ => (),
                }
            }
            
            // Early exit when max tokens - 1 reached (cause soft_stop will append eos) 
            if i >= (gen_config.max_tokens - 1) {
                finish_reason = FinishReason::Length;
                handler.soft_stop();
            }
        }

        if let Some(rest) = self.tokenizer.finalize_stream(&mut stream_ctx)? {
            callback(StreamFrame {
                output: &rest,
                tool_call_chunk: None,
                output_type: handler.gen_type(),
                logprobs: None,
            })?;
            content_text.push_str(&rest);
        }
        println!();

        let token_generation_sec = generation_start.elapsed().as_secs_f64();
        debug!("Generation complete in {:.3}s", token_generation_sec);
        let reasoning_tokens = handler.reasoning_token_count();
        Ok(
            GenerationReport::new(
                &content_text, 
                &reasoning_text,
                handler.tool_calls(),
                finish_reason,
                input_tokens_len as u32,
                i as u32,
                reasoning_tokens,
                token_generation_sec, 
            )
        )
    }

    fn match_cache_ctx(&mut self, prompt_token_ids: &[u32]) -> Result<usize, LociError> {
        let cache_seq_len_dim = self.model.cache_seq_len_dim();
        let min_prefill_tokens = self.model.min_prefill_tokens();
        let cache_len = self.generation_context.token_ids.len();
        let result = if self.prefix_caching {
            self.model_cache_manager.match_cache(&self.generation_context, &prompt_token_ids, cache_seq_len_dim, min_prefill_tokens, &self.device)?
        } else {
            MatchCacheResult::NoMatch
        };
        match result {
            MatchCacheResult::FullyMatchedActiveCache => {
                debug!("Matched cache length: {}", cache_len);
                Ok(cache_len)
            }
            MatchCacheResult::PartiallyMatchedActiveCache { matched_cache_length } => {
                for cache in &mut self.generation_context.cache {
                    if let Some(MixedCache::KvCache(concat_kv_cache)) = cache {
                        if concat_kv_cache.k().is_some() && concat_kv_cache.v().is_some() {
                            let mut k = concat_kv_cache.k_mut().unwrap();
                            *k = k.narrow(cache_seq_len_dim, 0, matched_cache_length)
                                .map_err(|e| LociError::CacheLoad(e.to_string()))?;
                            let mut v = concat_kv_cache.v_mut().unwrap();
                            *v = v.narrow(cache_seq_len_dim, 0, matched_cache_length)
                                .map_err(|e| LociError::CacheLoad(e.to_string()))?;
                        }
                    } else {
                        *cache = None;
                    }
                }

                debug!("Matched cache length: {}", matched_cache_length);
                self.generation_context.token_ids.truncate(matched_cache_length);
                Ok(matched_cache_length)
            }
            MatchCacheResult::MatchedInactiveCache { matched_cache_length, cache } => {
                debug!("Cache loaded from disk");
                debug!("Matched cache length: {}", matched_cache_length);
                self.generation_context.token_ids.clear();
                self.generation_context.token_ids.extend_from_slice(&prompt_token_ids[..matched_cache_length]);
                self.generation_context.cache = cache;
                Ok(matched_cache_length)
            }
            MatchCacheResult::NoMatch => {
                self.generation_context.token_ids.clear();
                self.generation_context.cache = self.model.init_cache()
                    .map_err(|e| LociError::ModelLoad(e.to_string()))?;
                Ok(0)
            },
        }
    }

    fn squeeze_logits(&self, logits: Tensor) -> anyhow::Result<Tensor> {
        let (_, seq_len, _) = logits.dims3()?;
        let last_token_logits = logits.narrow(1, seq_len - 1, 1)?;
        let squeezed = last_token_logits.squeeze(0)?.squeeze(0)?;

        anyhow::Ok(squeezed)
    }

    fn decode_sampling_result(&self, chosen_token: impl Into<String>, sampling_result: SamplingResult) -> Option<LogprobsContent> {
        let token = chosen_token.into();
        let logprob = sampling_result.logprob?;
        let bytes = token.as_bytes().to_vec();
        let top_logprobs = sampling_result.top_k_logprobs.and_then(|top_k_logprobs| top_k_logprobs.iter()
            .map(|top_k_entry| {
                let top_k_token = self.tokenizer.decode(&[top_k_entry.token_id], true)?;
                let top_k_logprob = top_k_entry.logprob;
                let top_k_bytes = top_k_token.as_bytes().to_vec();
                Ok(TopLogprobs {
                    token: top_k_token,
                    logprob: top_k_logprob,
                    bytes: top_k_bytes
                })
            })
            .collect::<Result<Vec<TopLogprobs>, LociError>>()
            .ok());

        Some(LogprobsContent {
            token,
            logprob,
            bytes,
            top_logprobs,
        })
    }

    pub fn flush_cache_to_file(&mut self) {
        if !self.prefix_caching { return; }
        if self.generation_context.token_ids.is_empty() { return; }
        if let Err(e) = self.model_cache_manager.save_cache(&self.generation_context) {
            warn!("Failed to flush cache: {e}");
        }
    }
}
