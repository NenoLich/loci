use std::mem::take;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::config::{
    GenerationConfig, GenerationConfigBuilder, GenerationOverrides, InferenceConfig, ModelConfig,
};
use crate::error::LociError;
use crate::gguf::Loader;
use crate::inference::{
    DeviceManager, GenerationContext, GenerationDataType, GenerationEvent, GenerationHandler,
    GenerationReport, InferenceSampler, PostSamplingConfig, ReasoningSupervisor,
    ReasoningSupervisorInterface, Sampler, SamplingResult, StopPatternMatcher, StreamCallback,
    StreamFrame, ToolCallingSupervisor, ToolCallingSupervisorInterface,
};
use crate::model::{Model, ModelBuilder};
use crate::profiling;
use crate::tokenizer::{StreamContext, Tokenizer, TokenizerService};
use crate::types::{
    ChatMessage, CompletionTokensDetails, FinishReason, LogprobsContent, ReasoningEffort, Tool,
    TopLogprobs, Usage,
};
use candle_core::{Device, Tensor};
use candle_transformers::quantized_var_builder::VarBuilder;
use memmap2::MmapOptions;
use tracing::{debug, debug_span};

#[derive(Default)]
pub struct InferenceEngineBuilder {
    gguf_path: Option<PathBuf>,
    inference_config: Option<InferenceConfig>,
    test_model: Option<Box<dyn Model + Send + Sync>>,
    test_tokenizer: Option<Box<dyn Tokenizer + Send + Sync>>,
}

impl InferenceEngineBuilder {
    pub fn with_gguf_metadata(mut self, path: impl AsRef<Path>) -> Self {
        self.gguf_path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn inference_config(mut self, inference_config: Option<InferenceConfig>) -> Self {
        self.inference_config = inference_config;
        self
    }

    #[cfg(any(test, feature = "mock"))]
    pub fn with_model(mut self, model: Box<dyn Model + Send + Sync>) -> Self {
        self.test_model = Some(model);
        self
    }

    #[cfg(any(test, feature = "mock"))]
    pub fn with_tokenizer(mut self, tokenizer: Box<dyn Tokenizer + Send + Sync>) -> Self {
        self.test_tokenizer = Some(tokenizer);
        self
    }

    fn init_model(
        &self,
        model_config: &ModelConfig,
        inference_config: &InferenceConfig,
        device: &Device,
    ) -> Result<Box<dyn Model + Send + Sync>, LociError> {
        let start_time = Instant::now();
        profiling::range_push!("VarBuilder Init");
        debug!("Creating VarBuilder...");
        let file = std::fs::File::open(model_config.file_path.clone())?;
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| LociError::ModelLoad(e.to_string()))?
        };
        let var_builder = debug_span!("VarBuilder init").in_scope(|| {
            VarBuilder::from_gguf_buffer(&mmap, device)
                .map_err(|e| LociError::ModelLoad(e.to_string()))
        })?;

        debug!("VarBuilder created");
        profiling::range_pop!();

        let model = debug_span!("Model load").in_scope(|| {
            ModelBuilder::new(model_config.clone(), var_builder, inference_config).build()
        })?;
        debug!("Model loaded in {:.3}s", start_time.elapsed().as_secs_f32());
        Ok(model)
    }

    pub fn build(self) -> Result<InferenceEngine, LociError> {
        let gguf_path = self
            .gguf_path
            .as_deref()
            .ok_or_else(|| LociError::ModelLoad("gguf_path is required but was not set".into()))?;
        let model_path = &gguf_path.to_string_lossy();
        let (_, model_name) = model_path
            .rsplit_once(['/', '\\'])
            .unwrap_or(("", model_path));

        let gguf_info = Loader::load_gguf_info(gguf_path, 0, false)?;

        let gen_builder = GenerationConfig::builder()
            .with_gguf_metadata(&gguf_info)
            .map_err(|e| LociError::ModelLoad(e.to_string()))?;

        let device = DeviceManager::select()?;

        let model_config = ModelConfig::from_gguf_info(&gguf_info)
            .map_err(|e| LociError::ModelLoad(format!("failed to parse model config: {}", e)))?;

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
            Some(ref config) => config,
            None => &InferenceConfig::default(),
        };
        let flash_attn = inference_config.flash_attn;

        let model = if let Some(model) = self.test_model {
            model
        } else {
            self.init_model(&model_config, inference_config, &device)?
        };

        profiling::range_push!("Tokenizer build");
        let tokenizer = if let Some(tokenizer) = self.test_tokenizer {
            tokenizer
        } else {
            let tokenizer_service = TokenizerService::builder()
                .with_gguf_metadata(&gguf_info)
                .build()?;
            Box::new(tokenizer_service) as Box<dyn Tokenizer + Send + Sync>
        };
        profiling::range_pop!();

        Ok(InferenceEngine {
            tokenizer,
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
    pub(crate) tokenizer: Box<dyn Tokenizer + Send + Sync>,
    pub(crate) device: Device,
    pub(crate) model_name: String,
    pub(crate) vocab_size: usize,
    pub(crate) model: Box<dyn Model + Send + Sync>,
    pub(crate) flash_attn: bool,
    pub(crate) supports_reasoning: bool,
    pub(crate) supports_tool_calling: bool,
    pub(crate) flatten_tools_to_functions: bool,
    pub(crate) post_sampling_config: PostSamplingConfig,
    pub(crate) gen_builder: GenerationConfigBuilder,
}

impl InferenceEngine {
    pub fn builder() -> InferenceEngineBuilder {
        InferenceEngineBuilder::default()
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
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
    ) -> Result<GenerationReport, LociError> {
        let gen_config = self.gen_builder.clone().with_overrides(overrides).build();
        debug!("Generation parameters: {:#?}", gen_config);
        debug!("Using flash attention: {}", self.flash_attn);
        let enable_thinking =
            self.supports_reasoning && gen_config.reasoning_effort != ReasoningEffort::None;
        let prompt = self.tokenizer.apply_chat_template(
            messages,
            tools,
            enable_thinking,
            self.flatten_tools_to_functions,
        )?;
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
    ) -> Result<GenerationReport, LociError> {
        let gen_config = self.gen_builder.clone().with_overrides(overrides).build();
        let prompt_token_ids = self.tokenizer.encode(prompt, true)?;
        self.generate_from_encoding(prompt_token_ids, ctx, gen_config, callback)
    }

    pub fn generate_from_encoding(
        &self,
        prompt_token_ids: Vec<u32>,
        ctx: &mut GenerationContext,
        gen_config: GenerationConfig,
        mut callback: StreamCallback,
    ) -> Result<GenerationReport, LociError> {
        // Tokenize prompt
        let input_tokens_len = prompt_token_ids.len();
        debug!("Input tokens length: {}", input_tokens_len);

        // Initialize sampler (handles temperature, top-p, etc.)
        let tool_start_token_id = self.post_sampling_config.tool_call_start_token_id;
        let special_token_ids = self.tokenizer.special_token_ids();
        let mut sampler: Box<dyn Sampler> = Box::new(InferenceSampler::new(
            gen_config.clone(),
            special_token_ids,
            self.vocab_size,
            20,
            tool_start_token_id,
        ));
        prompt_token_ids
            .iter()
            .for_each(|token| sampler.add_token(*token));

        let matched_cache_len = self.match_ctx_cache(ctx, &prompt_token_ids)?;
        let input_tokens = &prompt_token_ids[matched_cache_len..];
        debug!(input_len = input_tokens.len());
        let end_token = self.tokenizer.eos_token_id();
        let stream_ctx = crate::tokenizer::StreamContext::with_capacity(16);

        let generation_start = Instant::now();
        let reasoning_supervisor = ReasoningSupervisor::new(
            self.supports_reasoning,
            &gen_config.reasoning_effort,
            &self.post_sampling_config,
        );
        let reasoning_supervisor =
            reasoning_supervisor.map(|s| Box::new(s) as Box<dyn ReasoningSupervisorInterface>);

        let tool_calling_supervisor = ToolCallingSupervisor::new(
            self.supports_tool_calling,
            &self.post_sampling_config,
            &self.tokenizer,
        )
        .map_err(|e| LociError::Inference {
            source: e.into_boxed_dyn_error(),
        })?;
        let tool_choice_template =
            if let Some(tool_calling_supervisor) = tool_calling_supervisor.as_ref() {
                tool_calling_supervisor
                    .get_tool_choice_template(&self.tokenizer, &gen_config.tool_choice)
                    .map_err(|e| LociError::Inference {
                        source: e.into_boxed_dyn_error(),
                    })?
            } else {
                None
            };
        let tool_calling_supervisor = tool_calling_supervisor
            .map(|s| Box::new(s) as Box<dyn ToolCallingSupervisorInterface + '_>);
        let reasoning_budget = reasoning_supervisor
            .as_ref()
            .map_or(0, |rs| rs.get_reasoning_budget() as usize);

        let handler = GenerationHandler::new(
            input_tokens,
            sampler,
            reasoning_supervisor,
            tool_calling_supervisor,
            end_token,
            tool_choice_template,
        )
        .map_err(|e| LociError::Inference {
            source: e.into_boxed_dyn_error(),
        })?;

        let stop_pattern_matcher =
            StopPatternMatcher::new(gen_config.stop_tokens.clone(), &self.tokenizer);

        let mut state = GenerationState {
            handler,
            stop_pattern_matcher,
            stream_ctx,
            finish_reason: FinishReason::Stop,
            content_text: String::with_capacity(gen_config.max_tokens),
            reasoning_text: String::with_capacity(reasoning_budget),
            i: 0,
        };

        // Autoregressive generation loop
        loop {
            let should_stop = self.step_generation(&mut state, ctx, &mut callback, &gen_config)?;
            if should_stop {
                break;
            }
        }

        println!();

        let token_generation_sec = generation_start.elapsed().as_secs_f64();
        debug!("Generation complete in {:.3}s", token_generation_sec);
        let reasoning_tokens = state.handler.reasoning_tokens_count();
        let prompt_tokens = input_tokens_len as u32;
        let completion_tokens = state.i as u32;
        let usage = Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: None,
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens,
                ..Default::default()
            }),
        };

        Ok(GenerationReport::new(
            &state.content_text,
            &state.reasoning_text,
            state.handler.tool_calls(),
            state.finish_reason,
            usage,
            token_generation_sec,
        ))
    }

    fn step_generation(
        &self,
        state: &mut GenerationState,
        ctx: &mut GenerationContext,
        callback: &mut StreamCallback,
        gen_config: &GenerationConfig,
    ) -> Result<bool, LociError> {
        let token_span = debug_span!("generate_token");
        let _token_span_guard = token_span.enter();
        let input_tokens = take(&mut state.handler.input_tokens);
        let input = Tensor::new(input_tokens.as_slice(), &self.device)
            .map_err(|e| LociError::Inference {
                source: Box::new(e),
            })?
            .unsqueeze(0)
            .map_err(|e| LociError::Inference {
                source: Box::new(e),
            })?;
        let pos = ctx.token_ids.len();
        let logits = self
            .model
            .forward(&input, &mut ctx.active_cache, pos, self.flash_attn)
            .map_err(|e| LociError::Inference {
                source: e.into_boxed_dyn_error(),
            })?;
        let squeezed_logits = self
            .squeeze_logits(logits)
            .map_err(|e| LociError::Inference {
                source: e.into_boxed_dyn_error(),
            })?;

        ctx.update(input_tokens)?;
        let event = state
            .handler
            .advance(
                &squeezed_logits,
                gen_config.logprobs,
                gen_config.top_logprobs,
            )
            .map_err(|e| LociError::Inference {
                source: e.into_boxed_dyn_error(),
            })?;
        state.i += 1;

        match event {
            GenerationEvent::GenerationStopped => {
                return Ok(true);
            }
            GenerationEvent::ContentSampled { sampling_result } => {
                if state.stop_pattern_matcher.matches(sampling_result.token) {
                    state.finish_reason = FinishReason::Stop;
                    state.handler.soft_stop();
                }
                if let Some(output) = self
                    .tokenizer
                    .process_token_stream(&mut state.stream_ctx, sampling_result.token)?
                {
                    let logprobs = self.decode_sampling_result(&output, sampling_result);

                    callback(StreamFrame {
                        output: &output,
                        tool_call_chunk: None,
                        output_type: GenerationDataType::DirectContent,
                        logprobs,
                    })?;
                    state.content_text.push_str(&output);
                }
            }
            GenerationEvent::ReasoningSampled { sampling_result } => {
                if state.stop_pattern_matcher.matches(sampling_result.token) {
                    state.finish_reason = FinishReason::Stop;
                    state.handler.soft_stop();
                }
                if let Some(output) = self
                    .tokenizer
                    .process_token_stream(&mut state.stream_ctx, sampling_result.token)?
                {
                    let logprobs = self.decode_sampling_result(&output, sampling_result);

                    callback(StreamFrame {
                        output: &output,
                        tool_call_chunk: None,
                        output_type: GenerationDataType::Reasoning,
                        logprobs,
                    })?;
                    state.reasoning_text.push_str(&output);
                }
            }
            GenerationEvent::ToolCallNameChunk { chunk } => {
                callback(StreamFrame {
                    output: "",
                    tool_call_chunk: Some(chunk),
                    output_type: GenerationDataType::ToolCallName,
                    logprobs: None,
                })?;
                state.finish_reason = FinishReason::ToolCalls;
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

        if state.i >= (gen_config.max_tokens - 1) {
            state.finish_reason = FinishReason::Length;
            state.handler.soft_stop();
        }

        Ok(false)
    }

    fn match_ctx_cache(
        &self,
        ctx: &mut GenerationContext,
        prompt_token_ids: &[u32],
    ) -> Result<usize, LociError> {
        let min_prefill_tokens = self.model.min_prefill_tokens();
        let conv_on_cpu = self.model.conv_on_cpu();
        let matched_cache_len = ctx.match_cache(
            prompt_token_ids,
            min_prefill_tokens,
            &self.device,
            conv_on_cpu,
        )?;
        if matched_cache_len == 0 {
            let new_cache = self
                .model
                .init_cache()
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

    fn decode_sampling_result(
        &self,
        chosen_token: impl Into<String>,
        sampling_result: SamplingResult,
    ) -> Option<LogprobsContent> {
        let token = chosen_token.into();
        let logprob = sampling_result.logprob?;
        let bytes = token.as_bytes().to_vec();
        let top_logprobs = sampling_result.top_k_logprobs.and_then(|top_k_logprobs| {
            top_k_logprobs
                .iter()
                .map(|top_k_entry| {
                    let top_k_token = self.tokenizer.decode(&[top_k_entry.token_id], true)?;
                    let top_k_logprob = top_k_entry.logprob;
                    let top_k_bytes = top_k_token.as_bytes().to_vec();
                    Ok(TopLogprobs {
                        token: top_k_token,
                        logprob: top_k_logprob,
                        bytes: top_k_bytes,
                    })
                })
                .collect::<Result<Vec<TopLogprobs>, LociError>>()
                .ok()
        });

        Some(LogprobsContent {
            token,
            logprob,
            bytes,
            top_logprobs,
        })
    }
}

struct GenerationState<'a> {
    handler: GenerationHandler<'a>,
    stop_pattern_matcher: StopPatternMatcher,
    stream_ctx: StreamContext,
    finish_reason: FinishReason,
    content_text: String,
    reasoning_text: String,
    i: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::model_cache::{MockCacheLoader, MockModelCacheManagerInterface};
    use crate::inference::sampler::TopKEntry;
    use crate::model::model_base::{MixedCache, MockModel, ModelCacheType};
    use crate::tokenizer::MockTokenizer;
    use crate::types::ModelCacheFragmentation;
    use candle_nn::kv_cache::ConcatKvCache;
    use proptest::prelude::*;
    use rstest::rstest;

    fn vec3_strategy() -> impl Strategy<Value = Vec<Vec<Vec<f32>>>> {
        let element = -100.0f32..100.0;
        let dim3 = prop::collection::vec(element, 1..100);
        let dim2 = prop::collection::vec(dim3, 1);
        let dim1 = prop::collection::vec(dim2, 1);
        dim1
    }

    fn setup_test_engine(
        model: Box<dyn Model + Send + Sync>,
        tokenizer: Box<dyn Tokenizer + Send + Sync>,
    ) -> InferenceEngine {
        InferenceEngine {
            tokenizer,
            model,
            device: Device::Cpu,
            model_name: "mock".to_string(),
            vocab_size: 100,
            flash_attn: false,
            supports_reasoning: false,
            supports_tool_calling: false,
            flatten_tools_to_functions: true,
            post_sampling_config: PostSamplingConfig::default(),
            gen_builder: GenerationConfigBuilder::default(),
        }
    }

    proptest! {
        #[test]
        fn test_squeeze_logits_success(
            logits_vec in vec3_strategy()
        ) {
            let last_dim_len = logits_vec[0][0].len();
            let logits = Tensor::new(logits_vec.clone(), &Device::Cpu).expect(&format!("failed to create tensor from: {:#?}", logits_vec));
            let model = Box::new(MockModel::new());
            let tokenizer = Box::new(MockTokenizer::new());
            let engine = setup_test_engine(model, tokenizer);
            let result_tensor = engine.squeeze_logits(logits).expect("failed to squeeze logits");
            assert!(result_tensor.dims1().is_ok());
            let result_vec = result_tensor.to_vec1::<f32>().unwrap();
            assert_eq!(result_vec.len(), last_dim_len);
        }
    }

    #[test]
    fn test_squeeze_logits_failure_with_not_dims3_tensor() {
        let logits = Tensor::new(vec![vec![1.0, 2.0, 3.0]], &Device::Cpu).unwrap();
        let model = Box::new(MockModel::new());
        let tokenizer = Box::new(MockTokenizer::new());
        let engine = setup_test_engine(model, tokenizer);
        let result = engine.squeeze_logits(logits);
        assert!(result.is_err());
    }

    #[rstest]
    #[case(
        String::new(),
        1,
        Some(1.0),
        Some(vec![TopKEntry { token_id: 2, logprob: 0.5 }]),
        vec![String::from("foo")],
        Some(LogprobsContent {
            token: String::new(),
            logprob: 1.0,
            bytes: vec![],
            top_logprobs: Some(vec![TopLogprobs {
                token: String::from("foo"),
                logprob: 0.5,
                bytes: String::from("foo").as_bytes().to_vec(),
            }]),
        })
    )]
    #[case(
        String::from("some_token"),
        1,
        None,
        Some(vec![TopKEntry { token_id: 2, logprob: 0.5 }]),
        vec![String::from("foo")],
        None,
    )]
    #[case(
        String::from("some_token"),
        1,
        Some(1.0),
        None,
        vec![],
        Some(LogprobsContent {
            token: String::from("some_token"),
            logprob: 1.0,
            bytes: String::from("some_token").as_bytes().to_vec(),
            top_logprobs: None,
        })
    )]
    #[case(
        String::from("some_token"),
        1,
        Some(1.0),
        Some(vec![]),
        vec![],
        Some(LogprobsContent {
            token: String::from("some_token"),
            logprob: 1.0,
            bytes: String::from("some_token").as_bytes().to_vec(),
            top_logprobs: Some(vec![]),
        })
    )]
    #[case(
        String::from("some_token"),
        1,
        Some(1.0),
        Some(vec![TopKEntry { token_id: 2, logprob: 0.5 }, TopKEntry { token_id: 3, logprob: 0.7 }]),
        vec![String::from("foo"), String::from("bar")],
        Some(LogprobsContent {
            token: String::from("some_token"),
            logprob: 1.0,
            bytes: String::from("some_token").as_bytes().to_vec(),
            top_logprobs: Some(vec![TopLogprobs {
                token: String::from("foo"),
                logprob: 0.5,
                bytes: String::from("foo").as_bytes().to_vec(),
            }, TopLogprobs {
                token: String::from("bar"),
                logprob: 0.7,
                bytes: String::from("bar").as_bytes().to_vec(),
            }]),
        })
    )]
    fn test_decode_sampling_result(
        #[case] token: String,
        #[case] token_id: u32,
        #[case] logprob: Option<f32>,
        #[case] top_k_entrys: Option<Vec<TopKEntry>>,
        #[case] expect_decode_return: Vec<String>,
        #[case] expected: Option<LogprobsContent>,
    ) {
        let sampling_result = SamplingResult {
            token: token_id.clone(),
            logprob,
            top_k_logprobs: top_k_entrys,
        };
        let model = Box::new(MockModel::new());
        let mut tokenizer = MockTokenizer::new();
        let tokenizer_expect_decode_times_called =
            (!expect_decode_return.is_empty() && logprob.is_some()) as usize;
        for decode_return in expect_decode_return.into_iter() {
            tokenizer
                .expect_decode()
                .times(tokenizer_expect_decode_times_called)
                .returning(move |_, _| Ok(decode_return.clone()));
        }

        let engine = setup_test_engine(model, Box::new(tokenizer));
        let result = engine.decode_sampling_result(token, sampling_result);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_match_ctx_cache() {
        let mut model = MockModel::new();
        model.expect_conv_on_cpu().returning(|| true);
        model.expect_min_prefill_tokens().returning(|| 1);
        let tokenizer = Box::new(MockTokenizer::new());
        let engine = setup_test_engine(Box::new(model), tokenizer);
        let mut ctx = GenerationContext {
            model: "mock".to_string(),
            token_ids: vec![1, 2, 3, 4],
            active_cache: vec![
                Some(MixedCache::KvCache(ConcatKvCache::new(2))),
                Some(MixedCache::KvCache(ConcatKvCache::new(2))),
            ],
            model_layers_count: 2,
            model_cache_seq_len_dim: 2,
            model_cache_fragmentation: ModelCacheFragmentation::TokenWise,
            block_boundary_conv_cache: vec![],
            prefix_caching: true,
            cache_type: ModelCacheType::FullAttn,
            cache_manager: Box::new(MockModelCacheManagerInterface::new()),
            cache_loader: Box::new(MockCacheLoader::new()),
            cache_metadata: vec![],
        };
        let result = engine
            .match_ctx_cache(&mut ctx, &[1, 2, 3, 4, 5])
            .expect("failed to match ctx cache");
        assert_eq!(result, 4);
        assert_eq!(ctx.token_ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_match_ctx_cache_no_match_ctx_reset() {
        let mut model = MockModel::new();
        model.expect_conv_on_cpu().returning(|| true);
        model.expect_min_prefill_tokens().returning(|| 1);
        model.expect_init_cache().returning(|| {
            Ok(vec![
                Some(MixedCache::KvCache(ConcatKvCache::new(2))),
                Some(MixedCache::KvCache(ConcatKvCache::new(2))),
            ])
        });
        let tokenizer = Box::new(MockTokenizer::new());
        let engine = setup_test_engine(Box::new(model), tokenizer);
        let mut cache_manager = MockModelCacheManagerInterface::new();
        cache_manager.expect_min_cache_tokens().returning(|| 1);
        cache_manager
            .expect_save_cache()
            .returning(|_, _, _, _, _| Ok(PathBuf::from("test")));
        cache_manager.expect_enforce_limits().returning(|_| Ok(()));
        let mut ctx = GenerationContext {
            model: "mock".to_string(),
            token_ids: vec![1, 2, 3, 4],
            active_cache: vec![
                Some(MixedCache::KvCache(ConcatKvCache::new(2))),
                Some(MixedCache::KvCache(ConcatKvCache::new(2))),
            ],
            model_layers_count: 2,
            model_cache_seq_len_dim: 2,
            model_cache_fragmentation: ModelCacheFragmentation::TokenWise,
            block_boundary_conv_cache: vec![],
            prefix_caching: true,
            cache_type: ModelCacheType::FullAttn,
            cache_manager: Box::new(cache_manager),
            cache_loader: Box::new(MockCacheLoader::new()),
            cache_metadata: vec![],
        };
        let result = engine
            .match_ctx_cache(&mut ctx, &[0, 2, 3, 4, 5])
            .expect("failed to match ctx cache");
        assert_eq!(result, 0);
        assert_eq!(ctx.token_ids, Vec::<u32>::new());
    }
}
