use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::error::LociError;
use crate::gguf::Loader;
use crate::model::{MixedCache, Model, ModelBuilder};
use crate::config::ModelConfig;
use crate::tokenizer::{StreamState, TokenizerService, TokenizerServiceBuilder};
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::quantized_var_builder::VarBuilder;
use once_cell::sync::OnceCell;
use tracing::{debug};

pub struct GenerationReport {
    pub text: String,
    pub num_tokens: usize,
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
    model_path: Option<PathBuf>,
    dtype: DType,
}

impl InferenceEngineBuilder {
    pub fn new() -> Self {
        Self {
            model_path: None,
            dtype: DType::F16,
        }
    }

    pub fn model_path(mut self, path: impl AsRef<Path>) -> Self {
        self.model_path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn build(self) -> Result<InferenceEngine, LociError> {
        let model_path = self.model_path.ok_or_else(|| {
            LociError::ModelLoad("model_path is required but was not set".into())
        })?;

        let gguf_info = Loader::load_gguf_info(model_path.clone(), 0, false)?;

        let tokenizer = TokenizerServiceBuilder::from_gguf_metadata(&gguf_info.kv_meta)?;

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
        })
    }
}

pub struct InferenceEngine {
    tokenizer: TokenizerService,
    device: Device,
    config: ModelConfig,
    model: OnceCell<Box<dyn Model>>,
    compute_dtype: DType,
}

impl InferenceEngine {
    pub fn builder() -> InferenceEngineBuilder {
        InferenceEngineBuilder::new()
    }

    fn init_model(&self) -> anyhow::Result<Box<dyn Model>> {
        let start_time = Instant::now();
        let var_builder = VarBuilder::from_gguf(&self.config.file_path, &self.device)?;
        let model = ModelBuilder::new(self.config.clone(), var_builder, self.compute_dtype).build()?;
        debug!("Model loaded in {:.3}s", start_time.elapsed().as_secs_f32());
        anyhow::Ok(model)
    }

    pub fn generate_stream<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        use_flash: bool,
        mut callback: F,
    ) -> anyhow::Result<GenerationReport> 
    where F: FnMut(&str) -> anyhow::Result<()> 
    {
        // Tokenize prompt
        let encoding = self.tokenizer.encode(prompt)?;
        let prompt_tokens = encoding.get_ids();

        // Initialize logits processor (handles temperature, top-p, etc.)
        let logits_processor = LogitsProcessor::new(18, Some(temperature), None);

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
        let mut num_tokens = max_tokens;

        // Autoregressive generation loop
        for i in 1..max_tokens {
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
        debug!("Generation complete in {:.3}s", generation_start.elapsed().as_secs_f32());
        anyhow::Ok(GenerationReport { text: output_text, num_tokens })
    }

    fn generate_token(&self, context: &mut GenerationContext) -> anyhow::Result<u32> {
        let logits = self.forward(context)?;
        let squeezed_logits = self.squeeze_logits(logits)?;
        anyhow::Ok(context.logits_processor.sample(&squeezed_logits)?)
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
