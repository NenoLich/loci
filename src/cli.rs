use crate::inference::{InferenceEngine, StreamCallback};
use crate::gguf::Loader;
use crate::tokenizer::{TokenizerServiceBuilder, TokenizerService};
use crate::session::SessionManager;
use crate::config::{GenerationOverrides, InferenceConfig};
use crate::api::worker::EngineWorker;
use candle_core::DType;
use clap::{Parser, Subcommand, ValueEnum};
use std::rc::Rc;
use std::{ffi::OsString, str::FromStr};
use colored::*;

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
        #[arg(short = 'd', long = "dtype", value_enum, default_value_t = ComputeDtype::F32)]
        compute_dtype: ComputeDtype,
        #[arg(short = 'l', long = "max_seq_len", default_value_t = 32_000)]
        max_seq_len: usize,
        #[arg(short = 'f', long = "use_flash")]
        use_flash: bool,
        #[arg(short = 's', long = "stream")]
        stream: bool,
    },

    /// Generate text applying chat template to the prompt
    Chat {
        prompt: String,
        model_path: OsString,
        #[arg(long = "system_message", default_value = "You are a helpfull assistant.")]
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
        #[arg(short = 'd', long = "dtype", value_enum, default_value_t = ComputeDtype::F32)]
        compute_dtype: ComputeDtype,
        #[arg(short = 'l', long = "max_seq_len", default_value_t = 32_000)]
        max_seq_len: usize,
        #[arg(short = 'f', long = "use_flash")]
        use_flash: bool,
        #[arg(short = 's', long = "stream")]
        stream: bool,
    },

    /// Start the inference server
    Serve {
        #[arg(short = 'b', long = "bind", default_value = "127.0.0.1:8000")]
        bind: String,
        #[arg(short = 't', long = "timeout", default_value_t = 600)]
        idle_timeout: u64,
        #[arg(short = 'p', long = "model_path")]
        model_path: Option<OsString>,
        #[arg(short = 'd', long = "dtype", value_enum, default_value_t = ComputeDtype::F32)]
        compute_dtype: ComputeDtype,
        #[arg(short = 'l', long = "max_seq_len", default_value_t = 32_000)]
        max_seq_len: usize,
    }   
    
}

#[derive(Debug, Copy, Clone, ValueEnum)]
enum ComputeDtype {
    F32,
    F16
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

        Commands::Tokenize { text, model_path } => {
            let path_str = model_path.to_string_lossy();
            let path_sanitized = path_str.replace('\\', "/");
            let info = Loader::load_gguf_info(path_sanitized, 10, false)?;

            let mut tokenizer = TokenizerService::builder()
                .with_gguf_metadata(&info)
                .build()?;
            let encoding = tokenizer.encode(&text, true)?;

            println!("Input: \"{}\"", text);
            let tokens = encoding.get_ids();
            println!("Tokens: {:?}", tokens);
            let decoded = tokenizer.decode(encoding.get_ids(), false)?;
            println!("Decoded: \"{}\"", decoded.trim());
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
            use_flash,
            stream,
        } => {
            let inference_engine = init_inference_engine(model_path, compute_dtype, max_seq_len)?;

            let gen_overrides = GenerationOverrides::new(
                temperature,
                top_p,
                max_tokens,
                repetition_penalty,
                seed,
            );

            println!("🦀 Generating: \"{}\"", prompt);
            let start = std::time::Instant::now();

            let stream_callback: StreamCallback = if stream {
                Box::new(|output_chunk| {
                        print!("{}", output_chunk.green());
                        use std::io::Write;
                        std::io::stdout().flush()?;
                        anyhow::Ok(())
                    })
            } else {
                Box::new(|_| { anyhow::Ok(()) })
            };
            let report = inference_engine.generate_stream(
                prompt.as_str(),
                gen_overrides,
                use_flash,
                stream_callback,
            )?;

            if !stream {
                println!("\n✨ Output:\n{}", report.text);
            }

            let elapsed = start.elapsed();
            println!(
                "\n⏱️  Generated {} tokens in {:.2}s ({:.2} tok/s)",
                report.num_tokens,
                elapsed.as_secs_f64(),
                report.num_tokens as f64 / report.token_generation_sec,
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
            use_flash, 
            stream } => {
            let inference_engine = init_inference_engine(model_path, compute_dtype, max_seq_len)?;

            let gen_overrides = GenerationOverrides::new(
                temperature,
                top_p,
                max_tokens,
                repetition_penalty,
                seed,
            );

            println!("🦀 Generating: \"{}\"", prompt);
            let start = std::time::Instant::now();

            let stream_callback: StreamCallback = if stream {
                Box::new(|output_chunk| {
                        print!("{}", output_chunk.green());
                        use std::io::Write;
                        std::io::stdout().flush()?;
                        anyhow::Ok(())
                    })
            } else {
                Box::new(|_| { anyhow::Ok(()) })
            };

            let mut session_manager = SessionManager::new();
            let session = session_manager.start_session(&system_message);
            session.add_user_message(&prompt);
            let prompt_templated = session.get_messages();

            let report = inference_engine.generate_chat_stream(
                prompt_templated,
                &[],
                gen_overrides,
                use_flash,
                stream_callback,
            )?;

            let assistant_message = &report.text;
            session.add_assistant_message(assistant_message);
            let chat_messages = session.get_messages();

            println!("\n✨ Chat history:\n{:?}", chat_messages);

            let elapsed = start.elapsed();
            println!(
                "\n⏱️  Generated {} tokens in {:.2}s ({:.2} tok/s)",
                report.num_tokens,
                elapsed.as_secs_f64(),
                report.num_tokens as f64 / report.token_generation_sec,
            );
        },
        Commands::Serve { bind, idle_timeout, model_path, compute_dtype, max_seq_len } => {
            let inference_config = InferenceConfig::builder()
                .dtype(dtype)
                .max_seq_len(max_seq_len)
                .conv_on_cpu(true)
                .build();

            let engine_opt = model_path
                .and_then(|path| {
                    let path_str = model_path.to_string_lossy();
                    let path_sanitized = path_str.replace('\\', "/");
                    let engine = InferenceEngine::builder()
                        .with_gguf_metadata(&path_sanitized)
                        .config(inference_config)
                        .build()?;
                    Some(engine)
                });

            let (command_tx, command_rx) = tokio::sync::mpsc::channel(100);
            let worker = EngineWorker::new(inference_config, engine_opt, command_rx, idle_timeout);
            tokio::spawn(worker.run());
            
            api::run_server(command_tx, &bind).await?;
        },
    }

    anyhow::Ok(())
}

pub fn init_inference_engine(model_path: OsString, compute_dtype: ComputeDtype, max_seq_len: usize) -> anyhow::Result<InferenceEngine> {
    let path_str = model_path.to_string_lossy();
    let path_sanitized = path_str.replace('\\', "/");
    let dtype = match compute_dtype {
        ComputeDtype::F16 => DType::F16,
        ComputeDtype::F32 => DType::F32,
    };

    let inference_config = InferenceConfig::builder()
        .dtype(dtype)
        .max_seq_len(max_seq_len)
        .conv_on_cpu(true)
        .build();

    Ok(InferenceEngine::builder()
        .with_gguf_metadata(&path_sanitized)
        .config(inference_config)
        .build()?)
}
