use crate::inference::{InferenceEngine, GenerationContext, StreamCallback};
use crate::gguf::Loader;
use crate::tokenizer::{TokenizerServiceBuilder, TokenizerService, Tokenizer};
use crate::session::SessionManager;
use crate::config::{GenerationOverrides, InferenceConfig, ComputeDtype, FileConfig, InferenceFileConfig, ModelCacheConfig, CacheFileConfig};
use crate::api::run_server;
use crate::api::worker::EngineWorker;
use tokio_util::sync::CancellationToken;
use candle_core::DType;
use clap::{Parser, Subcommand, ValueEnum};
use std::rc::Rc;
use std::{ffi::OsString, str::FromStr};

#[derive(Parser)]
#[command(name = "loci")]
#[command(about = "Local LLM inference tool")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Info {
        path: OsString,

        /// Count of tensors to show
        #[arg(
            name = "first_k_tensors",
            short = 'k',
            long = "first_k",
            default_value_t = 10
        )]
        first_k_tensors: usize,
    },

    /// Tokenize text
    Tokenize {
        text: String,
        #[arg(short = 'p', long = "model_path")]
        model_path: OsString,
    },

    /// Decode tokens
    Decode {
        tokens: Vec<u32>,
        #[arg(short = 'p', long = "model_path")]
        model_path: OsString,
        #[arg(short = 's', long = "skip_special_tokens")]
        skip_special_tokens: bool,
    },

    /// Generate text from a prompt
    Generate {
        prompt: String,
        model_path: OsString,
        #[arg(short = 'm', long = "max_tokens")]
        max_tokens: Option<usize>,
        #[arg(short = 't', long = "temperature")]
        temperature: Option<f32>,
        #[arg(short = 'p', long = "top_p")]
        top_p: Option<f32>,
        #[arg(short = 'r', long = "repetition_penalty")]
        repetition_penalty: Option<f32>,
        #[arg(long = "seed")]
        seed: Option<usize>,
        #[arg(short = 'd', long = "dtype", value_enum)]
        compute_dtype: Option<ComputeDtype>,
        #[arg(short = 'l', long = "max_seq_len")]
        max_seq_len: Option<usize>,
        #[arg(long = "conv_on_cpu")]
        conv_on_cpu: Option<bool>,
        #[arg(short = 'f', long = "flash_attn")]
        flash_attn: Option<bool>,
        #[arg(short = 's', long = "stream")]
        stream: bool,
        #[arg(short = 'c', long = "config")]
        config_path: Option<OsString>,
    },

    /// Generate text applying chat template to the prompt
    Chat {
        prompt: String,
        model_path: OsString,
        #[arg(long = "system_message", default_value = "You are a helpful assistant.")]
        system_message: String,
        #[arg(short = 'm', long = "max_tokens")]
        max_tokens: Option<usize>,
        #[arg(short = 't', long = "temperature")]
        temperature: Option<f32>,
        #[arg(short = 'p', long = "top_p")]
        top_p: Option<f32>,
        #[arg(short = 'r', long = "repetition_penalty")]
        repetition_penalty: Option<f32>,
        #[arg(long = "seed")]
        seed: Option<usize>,
        #[arg(short = 'd', long = "dtype", value_enum)]
        compute_dtype: Option<ComputeDtype>,
        #[arg(short = 'l', long = "max_seq_len")]
        max_seq_len: Option<usize>,
        #[arg(long = "conv_on_cpu")]
        conv_on_cpu: Option<bool>,
        #[arg(short = 'f', long = "flash_attn")]
        flash_attn: Option<bool>,
        #[arg(short = 's', long = "stream")]
        stream: bool,
        #[arg(short = 'c', long = "config")]
        config_path: Option<OsString>,
    },

    /// Start the inference server
    Serve {
        #[arg(short = 'b', long = "bind", default_value = "127.0.0.1:8000")]
        bind: String,
        #[arg(short = 't', long = "timeout", default_value_t = 600)]
        idle_timeout: u64,
        #[arg(short = 'p', long = "model_path")]
        model_path: Option<OsString>,
        #[arg(short = 'd', long = "dtype", value_enum)]
        compute_dtype: Option<ComputeDtype>,
        #[arg(short = 'l', long = "max_seq_len")]
        max_seq_len: Option<usize>,
        #[arg(long = "conv_on_cpu")]
        conv_on_cpu: Option<bool>,
        #[arg(short = 'f', long = "flash_attn")]
        flash_attn: Option<bool>,
        #[arg(long = "cache_dir")]
        cache_dir: Option<OsString>,
        #[arg(long = "max_cache_size")]
        max_cache_size: Option<u64>,
        #[arg(long = "min_cache_tokens")]
        min_cache_tokens: Option<usize>,
        #[arg(long = "cache_block_size")]
        cache_block_size: Option<usize>,
        #[arg(long = "prefix_caching")]
        prefix_caching: Option<bool>,
        #[arg(short = 'c', long = "config")]
        config_path: Option<OsString>,
    }   
    
}

pub async fn run() -> anyhow::Result<()> {
    let args = Cli::parse();

    match args.command {
        Commands::Info {
            path,
            first_k_tensors,
        } => {
            let path_str = path.to_string_lossy();
            let path_sanitized = path_str.replace('\\', "/");
            let _info = Loader::load_gguf_info(&path_sanitized, first_k_tensors, true)?;
        }

        Commands::Tokenize { text, model_path} => {
            let path_str = model_path.to_string_lossy();
            let path_sanitized = path_str.replace('\\', "/");
            let info = Loader::load_gguf_info(path_sanitized, 10, false)?;

            let mut tokenizer = TokenizerService::builder()
                .with_gguf_metadata(&info)
                .build()?;
            let tokens = tokenizer.encode(&text, true)?;

            println!("Input: \"{}\"", text);
            println!("Tokens: {:?}", tokens);
            let decoded = tokenizer.decode(&tokens, false)?;
            println!("Decoded: \"{}\"", decoded.trim());
        }

        Commands::Decode { tokens, model_path, skip_special_tokens } => {
            let path_str = model_path.to_string_lossy();
            let path_sanitized = path_str.replace('\\', "/");
            let info = Loader::load_gguf_info(path_sanitized, 10, false)?;

            let mut tokenizer = TokenizerService::builder()
                .with_gguf_metadata(&info)
                .build()?;
            let decoded = tokenizer.decode(&tokens, skip_special_tokens)?; 
            println!("{}", decoded);
        }

        Commands::Generate {
            prompt,
            model_path,
            max_tokens,
            temperature,
            top_p,
            repetition_penalty,
            seed,
            compute_dtype,
            max_seq_len,
            conv_on_cpu,
            flash_attn,
            stream,
            config_path,
        } => {
            let start = std::time::Instant::now();
            let (mut inference_engine, mut ctx, gen_overrides, stream_callback) = setup_generation(
                model_path, 
                max_tokens, 
                temperature, 
                top_p, 
                repetition_penalty, 
                seed, 
                compute_dtype, 
                max_seq_len, 
                conv_on_cpu, 
                flash_attn, 
                stream, 
                config_path, 
            )?;

            let model_loading_time = start.elapsed().as_secs_f64();
            println!("🦀 Generating: \"{}\"", prompt);
            
            let report = inference_engine.generate_stream(
                prompt.as_str(),
                &mut ctx,
                gen_overrides,
                stream_callback,
            )?;

            if !stream {
                println!("\n✨ Output:\n{:?}", report.chat_message);
            }

            println!("\n⏱️  Model loading time: {:.2}s", model_loading_time);
            println!(
                "⏱️  Generated {} tokens in {:.2}s ({:.2} tok/s)",
                report.usage.completion_tokens,
                report.token_generation_sec,
                report.usage.completion_tokens as f64 / report.token_generation_sec,
            );
        },
        Commands::Chat { 
            prompt, 
            model_path, 
            system_message, 
            max_tokens, 
            temperature, 
            top_p, 
            repetition_penalty,
            seed, 
            compute_dtype, 
            max_seq_len, 
            conv_on_cpu,
            flash_attn,
            stream,
            config_path,
        } => {
            let start = std::time::Instant::now();
            let (mut inference_engine, mut ctx, gen_overrides, stream_callback) = setup_generation(
                model_path, 
                max_tokens, 
                temperature, 
                top_p, 
                repetition_penalty, 
                seed, 
                compute_dtype, 
                max_seq_len, 
                conv_on_cpu, 
                flash_attn, 
                stream, 
                config_path, 
            )?;

            let model_loading_time = start.elapsed().as_secs_f64();

            println!("🦀 Generating: \"{}\"", prompt);

            let mut session_manager = SessionManager::new();
            let session = session_manager.start_session(&system_message);
            session.add_user_message(&prompt);
            let prompt_templated = session.get_messages();

            let report = inference_engine.generate_chat_stream(
                prompt_templated,
                &[],
                &mut ctx,
                gen_overrides,
                stream_callback,
            )?;

            session.add_message(report.chat_message);
            let chat_messages = session.get_messages();

            println!("\n✨ Chat history:\n{:?}", chat_messages);

            println!("\n⏱️  Model loading time: {:.2}s", model_loading_time);
            println!(
                "⏱️  Generated {} tokens in {:.2}s ({:.2} tok/s)",
                report.usage.completion_tokens,
                report.token_generation_sec,
                report.usage.completion_tokens as f64 / report.token_generation_sec,
            );
        },
        Commands::Serve { 
            bind, 
            idle_timeout, 
            model_path, 
            compute_dtype, 
            max_seq_len,
            conv_on_cpu,
            flash_attn,
            cache_dir,
            max_cache_size,
            min_cache_tokens,
            cache_block_size,
            prefix_caching,
            config_path,
        } => {
            let mut inference_config_from_file = None;
            let mut model_cache_config_from_file = None;
            if let Some(config) = config_path {
                let config_from_file = FileConfig::load(&config.to_string_lossy().replace('\\', "/"))?;
                inference_config_from_file = config_from_file.inference_config;
                model_cache_config_from_file = config_from_file.cache_config;
            }

            let inference_config = build_inference_config(compute_dtype, max_seq_len, flash_attn, conv_on_cpu, inference_config_from_file);
            let model_cache_config = ModelCacheConfig::builder()
                .prefix_caching(prefix_caching)
                .cache_dir(cache_dir)
                .max_cache_size(max_cache_size)
                .min_cache_tokens(min_cache_tokens)
                .cache_block_size(cache_block_size)
                .with_file_config(model_cache_config_from_file)
                .build();

            let engine_opt = if let Some(model_path_str) = model_path {
                Some(init_inference_engine(model_path_str, Some(inference_config.clone()))?)
            } else {
                None
            };

            let cancelation_token = CancellationToken::new();
            let (command_tx, command_rx) = tokio::sync::mpsc::channel(100);
            let worker = EngineWorker::new(inference_config, model_cache_config, engine_opt, command_rx, idle_timeout, cancelation_token.clone())?;
            let worker_handle = tokio::spawn(worker.run());

            run_server(command_tx.clone(), &bind, cancelation_token).await?;
            drop(command_tx);

            worker_handle.await?;
        },
    }

    anyhow::Ok(())
}

pub fn setup_generation(
    model_path: OsString,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    seed: Option<usize>,
    compute_dtype: Option<ComputeDtype>,
    max_seq_len: Option<usize>,
    conv_on_cpu: Option<bool>,
    flash_attn: Option<bool>,
    stream: bool,
    config_path: Option<OsString>,
) -> anyhow::Result<(InferenceEngine, GenerationContext, GenerationOverrides, StreamCallback)> {
    let mut inference_config_from_file = None;
    let mut generation_config_from_file = None;
    if let Some(config) = config_path {
        let config_from_file = FileConfig::load(&config.to_string_lossy().replace('\\', "/"))?;
        inference_config_from_file = config_from_file.inference_config;
        generation_config_from_file = config_from_file.generation_config;
    }

    let inference_config = build_inference_config(compute_dtype, max_seq_len, flash_attn, conv_on_cpu, inference_config_from_file);
    let inference_engine = init_inference_engine(
        model_path, 
        Some(inference_config),
    )?;

    let gen_overrides = GenerationOverrides::new(
        temperature,
        top_p,
        max_tokens,
        repetition_penalty,
        None,
        None,
        None,
        None,
        None,
        seed,
        generation_config_from_file,
    );

    let stream_callback: StreamCallback = if stream {
        Box::new(crate::render::stdout_callback)
    } else {
        Box::new(|_| { Ok(()) })
    };

    let ctx = GenerationContext::new(&inference_engine.model_name(), None, inference_engine.model_cache_info())?;

    Ok((inference_engine, ctx, gen_overrides, stream_callback))
}

pub fn build_inference_config(compute_dtype: Option<ComputeDtype>, max_seq_len: Option<usize>, flash_attn: Option<bool>, conv_on_cpu: Option<bool>, config: Option<InferenceFileConfig>) -> InferenceConfig {
    InferenceConfig::builder()
        .dtype(compute_dtype)
        .max_seq_len(max_seq_len)
        .flash_attn(flash_attn)
        .conv_on_cpu(conv_on_cpu)
        .with_file_config(config)
        .build()
}

pub fn init_inference_engine(model_path: OsString, inference_config: Option<InferenceConfig>) -> anyhow::Result<InferenceEngine> {
    let path_str = model_path.to_string_lossy();
    let path_sanitized = path_str.replace('\\', "/");

    Ok(InferenceEngine::builder()
        .with_gguf_metadata(&path_sanitized)
        .inference_config(inference_config)
        .build()?
    )
}

