use std::path::{Path, PathBuf};
use std::time::Instant;
use std::mem::take;

use crate::error::LociError;
use crate::gguf::{GgufInfo, Loader};
use crate::model::{MixedCache, Model, ModelBuilder};
use crate::config::{ModelConfig, GenerationConfig, InferenceConfig, GenerationOverrides, GenerationConfigBuilder};
use crate::tokenizer::{StreamContext, TokenizerService, TokenizerServiceBuilder, Tokenizer};
use crate::types::{Role, ChatMessage, Tool, FinishReason, ReasoningEffort, Usage, CompletionTokensDetails, LogprobsContent, TopLogprobs, ChunkToolCall, ToolCall};
use crate::inference::{DeviceManager, InferenceSampler, Sampler, ReasoningSupervisor, ToolCallingSupervisor, StopPatternMatcher, SamplingResult, ToolFormatStyle, GenerationDataType, GenerationEvent, GenerationReport, StreamCallback, PostSamplingConfig, StreamFrame, GenerationHandler, GenerationContext};
use candle_core::{Device, Tensor};
use candle_transformers::quantized_var_builder::VarBuilder;
use memmap2::MmapOptions;
use tracing::{debug, debug_span};
use nvtx::{range_push, range_pop};

pub struct InferenceEngineBuilder {
    gguf_path: Option<PathBuf>,
    inference_config: Option<InferenceConfig>,
}

impl InferenceEngineBuilder {
    pub fn new() -> Self {
        Self {
            gguf_path: None,
            inference_config: None,
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

    fn init_model(&self, model_config: &ModelConfig, inference_config: &InferenceConfig, device: &Device) -> Result<Box<dyn Model + Send + Sync>, LociError> {
        let start_time = Instant::now();
        range_push!("VarBuilder Init");
        debug!("Creating VarBuilder...");
        let file = std::fs::File::open(model_config.file_path.clone())?;
        let mmap = unsafe {
            MmapOptions::new().map(&file)
                .map_err(|e| LociError::ModelLoad(e.to_string()))?
        };
        let var_builder = debug_span!("VarBuilder init").in_scope(||
            VarBuilder::from_gguf_buffer(&mmap, device)
                .map_err(|e| LociError::ModelLoad(e.to_string()))
        )?;
        
        debug!("VarBuilder created");
        range_pop!();

        let model = debug_span!("Model load").in_scope(||
            ModelBuilder::new(model_config.clone(), var_builder, inference_config).build()
        )?;
        debug!("Model loaded in {:.3}s", start_time.elapsed().as_secs_f32());
        Ok(model)
    }

    pub fn build(self) -> Result<InferenceEngine, LociError> {
        let gguf_path = self.gguf_path.as_deref().ok_or_else(|| {
            LociError::ModelLoad("gguf_path is required but was not set".into())
        })?;
        let model_path = &gguf_path.to_string_lossy();
        let (_, model_name) = model_path.rsplit_once(&['/', '\\']).unwrap_or(("", &model_path));

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
        let inference_config = match self.inference_config {
            Some(ref config) => &config,
            None => &InferenceConfig::default(),
        };
        let flash_attn = inference_config.flash_attn;

        let model = self.init_model(&model_config, inference_config, &device)?;

        Ok(InferenceEngine {
            tokenizer: Box::new(tokenizer) as Box<dyn Tokenizer + Send + Sync>,
            device,
            model_name: model_name.to_string(),
            vocab_size,
            model,
            flash_attn,
            supports_reasoning,
            supports_tool_calling,
            flatten_tools_to_functions: model_config.flatten_tools_to_functions,
            post_sampling_config,
            gen_builder,
        })
    }
}

pub struct InferenceEngine {
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
    device: Device,
    model_name: String,
    vocab_size: usize,
    model: Box<dyn Model + Send + Sync>,
    flash_attn: bool,
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

    pub fn model_name(&self) -> String {
        self.model_name.clone()
    }

    pub fn model_cache_info(&self) -> crate::model::ModelCacheInfo {
        self.model.cache_info()
    }

    pub fn generate_chat_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[Tool],
        ctx: &mut GenerationContext,
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
        let prompt_token_ids = self.tokenizer.encode(&prompt, false)?;
        self.generate_from_encoding(prompt_token_ids, ctx, gen_config, callback)
    }

    pub fn generate_stream(
        &self,
        prompt: &str,
        ctx: &mut GenerationContext,
        overrides: GenerationOverrides,
        callback: StreamCallback,
    ) -> Result<GenerationReport, LociError> 
    {
        let gen_config = self.gen_builder.clone().with_overrides(overrides).build();
        let prompt_token_ids = self.tokenizer.encode(&prompt, true)?;
        self.generate_from_encoding(prompt_token_ids, ctx, gen_config, callback)
    }

    pub fn generate_from_encoding(
        &self,
        prompt_token_ids: Vec<u32>,
        ctx: &mut GenerationContext,
        gen_config: GenerationConfig,
        mut callback: StreamCallback,
    ) -> Result<GenerationReport, LociError> 
    {
        // Tokenize prompt
        let input_tokens_len = prompt_token_ids.len();
        debug!("Input tokens length: {}", input_tokens_len);

        // Initialize sampler (handles temperature, top-p, etc.)
        let tool_start_token_id = self.post_sampling_config.tool_call_start_token_id;
        let special_token_ids = self.tokenizer.special_token_ids();
        let mut sampler: Box<dyn Sampler> = Box::new(InferenceSampler::new(gen_config.clone(), special_token_ids, self.vocab_size, 20, tool_start_token_id));
        prompt_token_ids.iter().for_each(|token| sampler.add_token(*token));

        let matched_cache_len = self.match_ctx_cache(
            ctx, 
            &prompt_token_ids,
        )?;
        let input_tokens = &prompt_token_ids[matched_cache_len..];
        debug!(input_len = input_tokens.len());
        let end_token = self.tokenizer.eos_token_id();
        let mut stream_ctx = crate::tokenizer::StreamContext::with_capacity(16);

        let generation_start = Instant::now();
        let mut reasoning_supervisor = ReasoningSupervisor::new(self.supports_reasoning, &gen_config.reasoning_effort, &self.post_sampling_config);
        let mut tool_calling_supervisor = ToolCallingSupervisor::new(self.supports_tool_calling, &self.post_sampling_config, &self.tokenizer)
            .map_err(|e| LociError::Inference{ source: e.into_boxed_dyn_error() })?;
        let mut tool_choice_template = if let Some(tool_calling_supervisor) = tool_calling_supervisor.as_ref() {
            tool_calling_supervisor.get_tool_choice_template(&self.tokenizer, &gen_config.tool_choice)
                .map_err(|e| LociError::Inference { source: e.into_boxed_dyn_error() })?
        } else {
            None
        };
        let reasoning_budget = reasoning_supervisor.as_ref().map_or(0, |rs| rs.reasoning_budget as usize);

        let mut handler = GenerationHandler::new(
            input_tokens, 
            sampler, 
            reasoning_supervisor, 
            tool_calling_supervisor,
            end_token,
            tool_choice_template,
        ).map_err(|e| LociError::Inference { source: e.into_boxed_dyn_error() })?;

        let mut stop_pattern_matcher = StopPatternMatcher::new(gen_config.stop_tokens.clone(), &self.tokenizer);

        let mut content_text = String::with_capacity(gen_config.max_tokens);
        let mut reasoning_text = String::with_capacity(reasoning_budget);
        let mut finish_reason = FinishReason::Stop;
        let mut i = 0;

        // Autoregressive generation loop
        loop {
            let should_stop = self.step_generation(
                &mut handler,
                ctx,
                &mut stop_pattern_matcher,
                &mut stream_ctx,
                &mut callback,
                &gen_config,
                &mut finish_reason,
                &mut content_text,
                &mut reasoning_text,
                &mut i,
            )?;
            if should_stop {
                break;
            }
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

    fn step_generation(
        &self,
        handler: &mut GenerationHandler,
        ctx: &mut GenerationContext,
        stop_pattern_matcher: &mut StopPatternMatcher,
        stream_ctx: &mut StreamContext,
        callback: &mut StreamCallback,
        gen_config: &GenerationConfig,
        finish_reason: &mut FinishReason,
        content_text: &mut String,
        reasoning_text: &mut String,
        i: &mut usize,
    ) -> Result<bool, LociError> {
        let token_span = debug_span!("generate_token");
        let _token_span_guard = token_span.enter();
        let input_tokens = take(&mut handler.input_tokens);
        let input = Tensor::new(input_tokens.as_slice(), &self.device)
            .map_err(|e| LociError::Inference{ source: Box::new(e) })?
            .unsqueeze(0)
            .map_err(|e| LociError::Inference{ source: Box::new(e) })?;
        let pos = ctx.token_ids.len();
        let logits = self.model.forward(&input, &mut ctx.active_cache, pos, self.flash_attn)
            .map_err(|e| LociError::Inference{ source: e.into_boxed_dyn_error() })?;
        let squeezed_logits = self.squeeze_logits(logits)
            .map_err(|e| LociError::Inference{ source: e.into_boxed_dyn_error() })?;

        ctx.update(input_tokens)?;
        let event = handler.advance(&squeezed_logits, gen_config.logprobs, gen_config.top_logprobs)
            .map_err(|e| LociError::Inference{ source: e.into_boxed_dyn_error() })?;
        *i += 1;

        match event {
            GenerationEvent::GenerationStopped => {
                return Ok(true);
            }
            GenerationEvent::ContentSampled { sampling_result } => {
                if stop_pattern_matcher.matches(sampling_result.token) {
                    *finish_reason = FinishReason::Stop;
                    handler.soft_stop();
                }
                if let Some(output) = self.tokenizer.process_token_stream(stream_ctx, sampling_result.token)? {
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
                    *finish_reason = FinishReason::Stop;
                    handler.soft_stop();
                }
                if let Some(output) = self.tokenizer.process_token_stream(stream_ctx, sampling_result.token)? {
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
                *finish_reason = FinishReason::ToolCalls;
            }
            GenerationEvent::ToolCallArgumentsChunk { chunk } => {
                callback(StreamFrame {
                    output: "",
                    tool_call_chunk: Some(chunk),
                    output_type: GenerationDataType::ToolCallArguments,
                    logprobs: None,
                })?;
            }
            _ => {}
        }

        if *i >= (gen_config.max_tokens - 1) {
            *finish_reason = FinishReason::Length;
            handler.soft_stop();
        }

        Ok(false)
    }

    fn match_ctx_cache(&self, ctx: &mut GenerationContext, prompt_token_ids: &[u32]) -> Result<usize, LociError> {
        let min_prefill_tokens = self.model.min_prefill_tokens();
        let conv_on_cpu = self.model.conv_on_cpu();
        let matched_cache_len = ctx.match_cache(prompt_token_ids, min_prefill_tokens, &self.device, conv_on_cpu)?;
        if matched_cache_len == 0 {
            let new_cache = self.model.init_cache()
                .map_err(|e| LociError::ModelLoad(e.to_string()))?;
            ctx.reset_active_cache(new_cache, true)?;
        } 
        Ok(matched_cache_len)
    }

    fn squeeze_logits(&self, logits: Tensor) -> anyhow::Result<Tensor> {
        let (_, seq_len, _) = logits.dims3()?;
        let last_token_logits = logits.narrow(1, seq_len - 1, 1)?;
        let squeezed = last_token_logits.squeeze(0)?.squeeze(0)?;

        // let squeezed_vec = squeezed.to_vec1::<f32>()?;
        // let mut squeezed_hm = squeezed_vec.iter().enumerate()
        //     .collect::<Vec<(usize, &f32)>>();
        // squeezed_hm.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        // let first_logits = squeezed_hm.iter().take(3)
        //     .map(|(id, logit)| format!("with id: {}, value: {}: {}", id, self.tokenizer.decode(&[*id as u32], false).unwrap(), logit))
        //     .collect::<Vec<String>>()
        //     .join(", ");
        // debug!("First 3 highest logits: {}", first_logits);

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
