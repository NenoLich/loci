use std::path::{Path, PathBuf};
use std::time::Instant;
use std::rc::Rc;

use crate::error::LociError;
use crate::gguf::{GgufInfo, Loader};
use crate::model::{MixedCache, Model, ModelBuilder};
use crate::config::{ModelConfig, GenerationConfig};
use crate::tokenizer::{StreamState, TokenizerService, TokenizerServiceBuilder};
use crate::session::ChatMessage;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::quantized_var_builder::VarBuilder;
use tokenizers::tokenizer::Encoding;
use memmap2::MmapOptions;
use once_cell::sync::OnceCell;
use tracing::{debug};
use nvtx::{range_push, range_pop};

pub struct GenerationReport {
    pub text: String,
    pub num_tokens: usize,
    pub token_generation_sec: f64,
}

struct GenerationContext {
    input_tokens: Vec<u32>,
    logits_processor: LogitsProcessor,
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
    gguf_info: Option<Rc<GgufInfo>>,
    dtype: DType,
    max_seq_len: usize,
    conv_on_cpu: bool,
}

impl InferenceEngineBuilder {
    pub fn new() -> Self {
        Self {
            gguf_info: None,
            dtype: DType::F16,
            max_seq_len: 32_000,
            conv_on_cpu: true,
        }
    }

    pub fn with_gguf_metadata(mut self, info: Rc<GgufInfo>) -> Self {
        self.gguf_info = Some(info);
        self
    }

    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    pub fn conv_on_cpu(mut self, conv_on_cpu: bool) -> Self {
        self.conv_on_cpu = conv_on_cpu;
        self
    }

    pub fn build(self) -> Result<InferenceEngine, LociError> {
        let gguf_info = self.gguf_info.ok_or_else(|| {
            LociError::ModelLoad("gguf_info is required but was not set".into())
        })?;
        range_push!("Tokenizer build");
        let tokenizer = TokenizerService::builder()
            .with_gguf_metadata(&gguf_info)
            .build()?;
        range_pop!();
        let device = DeviceManager::select()?;

        let config = ModelConfig::from_gguf_info(&gguf_info).map_err(|e| {
            LociError::ModelLoad(format!("failed to parse model config: {}", e))
        })?;

        Ok(InferenceEngine {
            tokenizer,
            device,
            config,
            model: OnceCell::new(),
            compute_dtype: self.dtype,
            max_seq_len: self.max_seq_len,
            conv_on_cpu: self.conv_on_cpu,
        })
    }
}

pub struct InferenceEngine {
    tokenizer: TokenizerService,
    device: Device,
    config: ModelConfig,
    model: OnceCell<Box<dyn Model>>,
    compute_dtype: DType,
    max_seq_len: usize,
    conv_on_cpu: bool,
}

impl InferenceEngine {
    pub fn builder() -> InferenceEngineBuilder {
        InferenceEngineBuilder::new()
    }

    fn init_model(&self) -> anyhow::Result<Box<dyn Model>> {
        let start_time = Instant::now();
        range_push!("VarBuilder Init");
        debug!("Creating VarBuilder...");
        let file = std::fs::File::open(&self.config.file_path)?;
        let mmap = unsafe {
            MmapOptions::new().map(&file)?
        };
        let var_builder = VarBuilder::from_gguf_buffer(&mmap, &self.device)?;

        debug!("VarBuilder created");
        range_pop!();
        let model = ModelBuilder::new(self.config.clone(), var_builder, self.compute_dtype, self.max_seq_len, self.conv_on_cpu).build()?;
        debug!("Model loaded in {:.3}s", start_time.elapsed().as_secs_f32());
        anyhow::Ok(model)
    }

    pub fn generate_chat_stream<F>(
        &mut self,
        messages: &[ChatMessage],
        gen_config: GenerationConfig,
        use_flash: bool,
        callback: F,
    ) -> anyhow::Result<GenerationReport> 
    where F: FnMut(&str) -> anyhow::Result<()> 
    {
        let prompt = self.tokenizer.apply_chat_template(messages)?;
        debug!("Model prompt: {:?}", prompt);
        let encoding = self.tokenizer.encode(&prompt, false)?;
        self.generate_from_encoding(encoding, gen_config, use_flash, callback)
    }

    pub fn generate_stream<F>(
        &mut self,
        prompt: &str,
        gen_config: GenerationConfig,
        use_flash: bool,
        callback: F,
    ) -> anyhow::Result<GenerationReport> 
    where F: FnMut(&str) -> anyhow::Result<()> 
    {
        let encoding = self.tokenizer.encode(&prompt, true)?;
        self.generate_from_encoding(encoding, gen_config, use_flash, callback)
    }

    pub fn generate_from_encoding<F>(
        &mut self,
        encoding: Encoding,
        gen_config: GenerationConfig,
        use_flash: bool,
        mut callback: F,
    ) -> anyhow::Result<GenerationReport> 
    where F: FnMut(&str) -> anyhow::Result<()> 
    {
        // Tokenize prompt
        let prompt_tokens = encoding.get_ids();
        let input_tokens_len = prompt_tokens.len();
        debug!("Input tokens length: {}", input_tokens_len);

        // Initialize logits processor (handles temperature, top-p, etc.)
        let logits_processor = LogitsProcessor::new(gen_config.seed, Some(gen_config.temperature), Some(gen_config.top_p));

        let model = self.model.get_or_try_init(|| self.init_model())?;

        let cache = model.init_cache()?;

        let end_token = self.tokenizer.eos_token_id();
        let mut stream_state = StreamState::default();

        let generation_start = Instant::now();
        let mut context = GenerationContext {
            input_tokens: prompt_tokens.to_vec(), 
            logits_processor, 
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
        let next_token = context.logits_processor.sample(&squeezed_logits)?;
        anyhow::Ok(next_token)
    }

    fn forward(
        &self,
        context: &mut GenerationContext,
    ) -> anyhow::Result<Tensor> {
        let model = self.model.get_or_try_init(|| self.init_model())?;
        let input = Tensor::new(context.input_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        model.forward(&input, &mut context.cache, context.pos, context.use_flash)
    }

    fn squeeze_logits(&self, logits: Tensor) -> anyhow::Result<Tensor> {
        let (_, seq_len, _) = logits.dims3()?;

        let last_token_logits = logits.narrow(1, seq_len - 1, 1)?.flatten_all()?;

        anyhow::Ok(last_token_logits)
    }
}
