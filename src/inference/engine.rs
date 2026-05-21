use std::path::{Path, PathBuf};
use std::time::Instant;
use std::rc::Rc;

use crate::error::LociError;
use crate::gguf::{GgufInfo, Loader};
use crate::model::{MixedCache, Model, ModelBuilder};
use crate::config::{ModelConfig, GenerationConfig, InferenceConfig, GenerationOverrides, GenerationConfigBuilder};
use crate::tokenizer::{StreamState, TokenizerService, TokenizerServiceBuilder};
use crate::api::types::{ChatMessage, Tool, FinishReason};
use crate::inference::InferenceSampler;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::quantized_var_builder::VarBuilder;
use tokenizers::tokenizer::Encoding;
use memmap2::MmapOptions;
use once_cell::sync::OnceCell;
use tracing::{debug};
use nvtx::{range_push, range_pop};

pub type StreamCallback = Box<dyn FnMut(&str) -> anyhow::Result<()>>;

pub enum GeneratedDataType {
    Content,
    Reasoning,
    ToolCall,
}

pub struct StreamFrame {
    output: String,
    logprobs: Option<Vec<f64>>,
}

pub struct GenerationReport {
    pub text: String,
    pub num_tokens: usize,
    pub token_generation_sec: f64,
}

struct GenerationContext {
    input_tokens: Vec<u32>,
    sampler: &mut InferenceSampler,
    cache: Vec<Option<MixedCache>>,
    pos: usize,
    use_flash: bool,
}

impl GenerationContext {
    fn set_input_tokens(&mut self, input_tokens: &[u32]) {
        self.input_tokens = input_tokens.to_vec();
    }

    fn advance(&mut self, new_input: &[u32]) {
        self.pos += self.input_tokens.len();
        self.set_input_tokens(new_input);
    }
}

pub struct DeviceManager;

impl DeviceManager {
    pub fn select() -> Result<Device, LociError> {
        if cfg!(feature = "cuda") && candle_core::utils::cuda_is_available() {
            debug!("Running on CUDA");
            Ok(Device::new_cuda(0).map_err(|e| LociError::ModelLoad(format!("CUDA device selection failed: {}", e)))?)
        } else {
            debug!("Running on CPU");
            Ok(Device::Cpu)
        }
    }
}

pub struct InferenceEngineBuilder {
    gguf_path: Option<PathBuf>,
    config: Option<InferenceConfig>,
}

impl InferenceEngineBuilder {
    pub fn new() -> Self {
        Self {
            gguf_path: None,
            config: None,
        }
    }

    pub fn with_gguf_metadata(mut self, path: impl AsRef<Path>) -> Self {
        self.gguf_path = Some(PathBuf::from(path));
        self
    }

    pub fn config(mut self, config: InferenceConfig) -> Self {
        self.config = Some(config);
        self
    }

    fn init_model(&self, inference_config: InferenceConfig, model_config: ModelConfig, device: &Device) -> anyhow::Result<Box<dyn Model>> {
        let start_time = Instant::now();
        range_push!("VarBuilder Init");
        debug!("Creating VarBuilder...");
        let file = std::fs::File::open(&model_config.file_path)?;
        let mmap = unsafe {
            MmapOptions::new().map(&file)?
        };
        let var_builder = VarBuilder::from_gguf_buffer(&mmap, device)?;

        debug!("VarBuilder created");
        range_pop!();
        let model = ModelBuilder::new(model_config.clone(), var_builder, inference_config).build()?;
        debug!("Model loaded in {:.3}s", start_time.elapsed().as_secs_f32());
        anyhow::Ok(model)
    }

    pub fn build(self) -> Result<InferenceEngine, LociError> {
        let gguf_path = self.gguf_path.ok_or_else(|| {
            LociError::ModelLoad("gguf_path is required but was not set".into())
        })?;

        let gguf_info = Loader::load_gguf_info(&gguf_path, 0, false)?;

        let gen_builder = GenerationConfig::builder()
            .with_gguf_metadata(&gguf_info)?;

        let inference_config = self.config.unwrap_or_default();
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
        let model = self.init_model(inference_config, model_config, &device)?;

        Ok(InferenceEngine {
            tokenizer,
            device,
            model_path: gguf_path.to_string_lossy().into(),
            vocab_size,
            model,
            gen_builder,
        })
    }
}

pub struct InferenceEngine {
    tokenizer: TokenizerService,
    device: Device,
    model_path: String,
    vocab_size: usize,
    model: Box<dyn Model>,
    gen_builder: GenerationConfigBuilder,
}

impl InferenceEngine {
    pub fn builder() -> InferenceEngineBuilder {
        InferenceEngineBuilder::new()
    }

    pub fn model_path(&self) -> String {
        self.model_path.clone()
    }

    pub fn generate_chat_stream<F>(
        &self,
        messages: &[ChatMessage],
        tools: &[Tool],
        overrides: GenerationOverrides,
        use_flash: bool,
        callback: StreamCallback,
    ) -> anyhow::Result<GenerationReport> 
    {
        let prompt = self.tokenizer.apply_chat_template(messages, tools)?;
        debug!("Model prompt: {:?}", prompt);
        let gen_config = self.gen_builder.with_overrides(overrides).build();
        let encoding = self.tokenizer.encode(&prompt, false)?;
        self.generate_from_encoding(encoding, gen_config, use_flash, callback)
    }

    pub fn generate_stream<F>(
        &self,
        prompt: &str,
        overrides: GenerationOverrides,
        use_flash: bool,
        callback: StreamCallback,
    ) -> anyhow::Result<GenerationReport> 
    {
        let gen_config = self.gen_builder.with_overrides(overrides).build();
        let encoding = self.tokenizer.encode(&prompt, true)?;
        self.generate_from_encoding(encoding, gen_config, use_flash, callback)
    }

    pub fn generate_from_encoding<F>(
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
        let mut sampler = InferenceSampler::new(gen_config.clone(), self.vocab_size, 20);
        prompt_tokens.iter().for_each(|&token| sampler.add_token(token));

        let cache = self.model.init_cache()?;

        let end_token = self.tokenizer.eos_token_id();
        let mut stream_state = StreamState::default();

        let generation_start = Instant::now();
        let mut context = GenerationContext {
            input_tokens: prompt_tokens.to_vec(), 
            sampler: &mut sampler, 
            cache, 
            pos: 0, 
            use_flash 
        };
        let mut next_token;
        let mut output_text = String::new();
        let mut num_tokens = gen_config.max_tokens;

        // Autoregressive generation loop
        for i in 1..gen_config.max_tokens {
            // This handles pre-fill on i=1, and single token generation on i>1
            next_token = self.generate_token(&mut context)?;

            // After pre-fill, we only ever feed the 'next_token' back in
            context.advance(&[next_token]);

            if let Some(output) = self.tokenizer.process_token(&mut stream_state, next_token)? {
                callback(&output)?;
                output_text.push_str(&output);
            }

            if next_token == end_token {
                num_tokens = i;
                break;
            }
        }

        if let Some(rest) = self.tokenizer.decode_rest(&mut stream_state)? {
            callback(&rest)?;
            output_text.push_str(&rest);
        }
        println!();

        let token_generation_sec = generation_start.elapsed().as_secs_f64();
        debug!("Generation complete in {:.3}s", token_generation_sec);
        anyhow::Ok(GenerationReport { text: output_text, num_tokens, token_generation_sec })
    }

    fn generate_token(&self, context: &mut GenerationContext) -> anyhow::Result<u32> {
        let logits = self.forward(context)?;
        let squeezed_logits = self.squeeze_logits(logits)?;
        let next_token = context.sampler.sample(&squeezed_logits)?;
        anyhow::Ok(next_token)
    }

    fn forward(
        &self,
        context: &mut GenerationContext,
    ) -> anyhow::Result<Tensor> {
        let input = Tensor::new(context.input_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        self.model.forward(&input, &mut context.cache, context.pos, context.use_flash)
    }

    fn squeeze_logits(&self, logits: Tensor) -> anyhow::Result<Tensor> {
        let (_, seq_len, _) = logits.dims3()?;

        let last_token_logits = logits.narrow(1, seq_len - 1, 1)?.flatten_all()?;

        anyhow::Ok(last_token_logits)
    }
}
