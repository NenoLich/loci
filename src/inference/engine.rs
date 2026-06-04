use std::path::{Path, PathBuf};
use std::time::Instant;
use std::rc::Rc;

use crate::error::LociError;
use crate::gguf::{GgufInfo, Loader};
use crate::model::{MixedCache, Model, ModelBuilder};
use crate::config::{ModelConfig, GenerationConfig, InferenceConfig, GenerationOverrides, GenerationConfigBuilder};
use crate::tokenizer::{StreamContext, TokenizerService, TokenizerServiceBuilder};
use crate::api::types::{Role, ChatMessage, Tool, FinishReason, ReasoningEffort, Usage, CompletionTokensDetails, LogprobsContent, TopLogprobs, ChunkToolCall, ToolCall};
use crate::inference::{DeviceManager, InferenceSampler, ReasoningSupervisor, ToolCallingSupervisor, StopPatternMatcher, SamplingResult, ToolFormatStyle, GenerationDataType, GenerationEvent, GenerationReport, StreamCallback, PostSamplingConfig, StreamFrame, GenerationHandler};
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::quantized_var_builder::VarBuilder;
use tokenizers::tokenizer::Encoding;
use memmap2::MmapOptions;
use once_cell::sync::OnceCell;
use tracing::{debug};
use nvtx::{range_push, range_pop};

pub struct InferenceEngineBuilder<'a> {
    gguf_path: Option<PathBuf>,
    config: Option<&'a InferenceConfig>,
}

impl<'a> InferenceEngineBuilder<'a> {
    pub fn new() -> Self {
        Self {
            gguf_path: None,
            config: None,
        }
    }

    pub fn with_gguf_metadata(mut self, path: impl AsRef<Path>) -> Self {
        self.gguf_path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn config(mut self, config: &'a InferenceConfig) -> Self {
        self.config = Some(config);
        self
    }

    fn init_model(&self, model_config: &ModelConfig, device: &Device) -> Result<Box<dyn Model + Send + Sync>, LociError> {
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
        let inference_config = match self.config.as_ref() {
            Some(&config) => config,
            None => &InferenceConfig::default(),
        };
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
        let model = self.init_model(&model_config, &device)?;

        Ok(InferenceEngine {
            tokenizer,
            device,
            model_path: gguf_path.to_string_lossy().into(),
            vocab_size,
            model,
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
    device: Device,
    model_path: String,
    vocab_size: usize,
    model: Box<dyn Model + Send + Sync>,
    supports_reasoning: bool,
    supports_tool_calling: bool,
    flatten_tools_to_functions: bool,
    post_sampling_config: PostSamplingConfig,
    gen_builder: GenerationConfigBuilder,
}

impl InferenceEngine {
    pub fn builder<'a>() -> InferenceEngineBuilder<'a> {
        InferenceEngineBuilder::new()
    }

    pub fn model_path(&self) -> String {
        self.model_path.clone()
    }

    pub fn generate_chat_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[Tool],
        overrides: GenerationOverrides,
        use_flash: bool,
        callback: StreamCallback,
    ) -> anyhow::Result<GenerationReport> 
    {
        let gen_config = self.gen_builder.clone().with_overrides(overrides).build();
        debug!("Generation parameters: {:#?}", gen_config);
        debug!("Using flash attention: {}", use_flash);
        let enable_thinking = self.supports_reasoning && gen_config.reasoning_effort != ReasoningEffort::None;
        let prompt = self.tokenizer.apply_chat_template(messages, tools, enable_thinking, self.flatten_tools_to_functions)?;
        debug!("Model prompt: {:#?}", prompt); 
        let encoding = self.tokenizer.encode(&prompt, false)?;
        self.generate_from_encoding(encoding, gen_config, use_flash, callback)
    }

    pub fn generate_stream(
        &self,
        prompt: &str,
        overrides: GenerationOverrides,
        use_flash: bool,
        callback: StreamCallback,
    ) -> anyhow::Result<GenerationReport> 
    {
        let gen_config = self.gen_builder.clone().with_overrides(overrides).build();
        let encoding = self.tokenizer.encode(&prompt, true)?;
        self.generate_from_encoding(encoding, gen_config, use_flash, callback)
    }

    pub fn generate_from_encoding(
        &self,
        encoding: Encoding,
        gen_config: GenerationConfig,
        use_flash: bool,
        mut callback: StreamCallback,
    ) -> anyhow::Result<GenerationReport> 
    {
        // Tokenize prompt
        let prompt_tokens = encoding.get_ids();
        let input_tokens_len = prompt_tokens.len();
        debug!("Input tokens length: {}", input_tokens_len);

        // Initialize sampler (handles temperature, top-p, etc.)
        let tool_start_token_id = self.post_sampling_config.tool_call_start_token_id;
        let mut sampler = InferenceSampler::new(gen_config.clone(), self.vocab_size, 20, tool_start_token_id);
        
        let cache = self.model.init_cache()?;

        let end_token = self.tokenizer.eos_token_id();
        let mut stream_ctx = crate::tokenizer::StreamContext::with_capacity(8);

        let generation_start = Instant::now();
        let mut reasoning_supervisor = ReasoningSupervisor::new(self.supports_reasoning, &gen_config.reasoning_effort, &self.post_sampling_config);
        let mut tool_calling_supervisor = ToolCallingSupervisor::new(self.supports_tool_calling, &self.post_sampling_config, &self.tokenizer, &gen_config.tool_choice)?;
        let reasoning_budget = reasoning_supervisor.as_ref().map_or(0, |rs| rs.reasoning_budget as usize);

        let mut handler = GenerationHandler::new(
            prompt_tokens, 
            sampler, 
            reasoning_supervisor, 
            tool_calling_supervisor,
            cache, 
            end_token,
            use_flash
        );

        let mut stop_pattern_matcher = StopPatternMatcher::new(gen_config.stop_tokens, &self.tokenizer);

        let mut content_text = String::with_capacity(gen_config.max_tokens);
        let mut reasoning_text = String::with_capacity(reasoning_budget);
        let mut finish_reason = FinishReason::Stop;
        let mut i = 0;

        // Autoregressive generation loop
        'generation:loop {
            // This handles pre-fill on i=0, and single token generation on i>1
            self.generate_token(&mut handler, gen_config.logprobs, gen_config.top_logprobs)?;
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
        anyhow::Ok(
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

    fn generate_token(&self, handler: &mut GenerationHandler, with_logprobs: bool, top_k_logprobs: Option<usize>) -> anyhow::Result<()> {
        let logits = self.forward(handler)?;
        let squeezed_logits = self.squeeze_logits(logits)?;
        handler.advance(&squeezed_logits, with_logprobs, top_k_logprobs);
        Ok(())
    }

    fn forward(
        &self,
        handler: &mut GenerationHandler,
    ) -> anyhow::Result<Tensor> {
        let input = Tensor::new(handler.input_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        self.model.forward(&input, &mut handler.cache, handler.pos, handler.use_flash)
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
}
