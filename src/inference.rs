use std::any;
use std::time::Instant;

use crate::gguf_types::GgufInfo;
use crate::model::{MixedCache, Model, ModelBuilder};
use crate::model_config::ModelConfig;
use crate::tokenizer::{StreamState, TokenizerService, TokenizerServiceBuilder};
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::quantized_var_builder::VarBuilder;
use once_cell::sync::OnceCell;

pub struct InferenceEngine {
    tokenizer: TokenizerService,
    device: Device,
    config: ModelConfig,
    model: OnceCell<Box<dyn Model>>,
    compute_dtype: DType,
    use_flash: bool,
}

impl InferenceEngine {
    pub fn new(gguf_info: &GgufInfo, compute_dtype: DType, use_flash: bool) -> anyhow::Result<Self> {
        // Load tokenizer
        let gguf_meta = &gguf_info.kv_meta;
        let tokenizer = TokenizerServiceBuilder::from_gguf_metadata(gguf_meta)?;

        // Extract config from metadata
        let config = ModelConfig::from_gguf_info(gguf_info)?;

        // Choose device: CUDA if available, else CPU
        let device = if cfg!(feature = "cuda") && candle_core::utils::cuda_is_available() {
            println!("Running on CUDA");
            Device::new_cuda(0)?
        } else {
            println!("Running on CPU");
            Device::Cpu
        };

        anyhow::Ok(Self {
            tokenizer,
            device,
            config,
            model: OnceCell::new(),
            compute_dtype,
            use_flash,
        })
    }

    fn init_model(&self) -> anyhow::Result<Box<dyn Model>> {
        let start_time = Instant::now();
        let var_builder = VarBuilder::from_gguf(&self.config.file_path, &self.device)?;
        let model = ModelBuilder::new(self.config.clone(), var_builder, self.compute_dtype).build()?;
        println!("Model loaded in {:.3}s", start_time.elapsed().as_secs_f32());
        anyhow::Ok(model)
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
    ) -> anyhow::Result<String> {
        // Tokenize prompt
        let encoding = self.tokenizer.encode(prompt)?;
        let mut tokens = encoding.get_ids().to_vec();

        // Initialize logits processor (handles temperature, top-p, etc.)
        let mut logits_processor = LogitsProcessor::new(19, Some(temperature), None);

        let model = self.model.get_or_try_init(|| self.init_model())?;

        let mut cache = model.init_cache()?;

        // Pre-fil generation
        let generation_start = Instant::now();
        let input = Tensor::new(tokens.clone(), &self.device)?.unsqueeze(0)?;

        let mut pos = 0;
        let logits = self.forward(&input, &mut cache, pos, self.compute_dtype)?;
        pos = tokens.len();
        let squeezed_logits = self.squeeze_logits(logits)?;
        let mut next_token = logits_processor.sample(&squeezed_logits)?;
        println!("Selected token: {}", next_token);
        tokens.push(next_token);

        // Autoregressive generation loop
        for i in 1..max_tokens {
            println!("Step {}: Input Token ID: {}, Pos: {}", i, next_token, pos);
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;

            let logits = self.forward(&input, &mut cache, pos, self.compute_dtype)?;
            pos += 1;
            let squeezed_logits = self.squeeze_logits(logits)?;
            next_token = logits_processor.sample(&squeezed_logits)?;
            println!("Selected token: {}", next_token);
            tokens.push(next_token);

            if next_token == self.tokenizer.eos_token_id() {
                break;
            }
        }

        let decoded = self.tokenizer.decode(&tokens);
        println!("Generation complete in {:.3}s", generation_start.elapsed().as_secs_f32());
        decoded
    }

    pub fn generate_stream<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        mut callback: F,
    ) -> anyhow::Result<()> 
    where F: FnMut(String) -> anyhow::Result<()> 
    {
        // Tokenize prompt
        let encoding = self.tokenizer.encode(prompt)?;
        let mut tokens = encoding.get_ids().to_vec();

        // Initialize logits processor (handles temperature, top-p, etc.)
        let mut logits_processor = LogitsProcessor::new(19, Some(temperature), None);

        let model = self.model.get_or_try_init(|| self.init_model())?;

        let mut cache = model.init_cache()?;

        let end_token = self.tokenizer.eos_token_id();
        let mut stream_state = StreamState::default();

        // Pre-fil generation
        let generation_start = Instant::now();
        let input = Tensor::new(tokens.clone(), &self.device)?.unsqueeze(0)?;

        let mut pos = 0;
        let logits = self.forward(&input, &mut cache, pos, self.compute_dtype)?;
        pos = tokens.len();
        let squeezed_logits = self.squeeze_logits(logits)?;
        let mut next_token = logits_processor.sample(&squeezed_logits)?;

        if let Some(output) = self.tokenizer.process_token(&mut stream_state, next_token.clone())? {
            callback(output)?;
        }
        tokens.push(next_token);

        // Autoregressive generation loop
        for i in 1..max_tokens {

            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;

            let logits = self.forward(&input, &mut cache, pos, self.compute_dtype)?;
            pos += 1;
            let squeezed_logits = self.squeeze_logits(logits)?;
            next_token = logits_processor.sample(&squeezed_logits)?;

            if let Some(output) = self.tokenizer.process_token(&mut stream_state, next_token.clone())? {
                callback(output)?;
            }
            tokens.push(next_token);

            if next_token == end_token {
                break;
            }
        }

        if let Some(rest) = self.tokenizer.decode_rest(&mut stream_state)? {
            callback(rest)?;
        }
        println!();
        println!("Generation complete in {:.3}s", generation_start.elapsed().as_secs_f32());
        anyhow::Ok(())
    }

    fn forward(
        &self,
        input: &Tensor,
        cache: &mut Vec<Option<MixedCache>>,
        pos: usize,
        compute_dtype: DType,
    ) -> anyhow::Result<Tensor> {
        let model = self.model.get_or_try_init(|| self.init_model())?;
        model.forward(input, cache, pos, self.use_flash)
    }

    fn squeeze_logits(&self, logits: Tensor) -> anyhow::Result<Tensor> {
        let (_, seq_len, _) = logits.dims3()?;

        let last_token_logits = logits.narrow(1, seq_len - 1, 1)?.flatten_all()?;

        anyhow::Ok(last_token_logits)
    }
}
