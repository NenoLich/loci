use crate::inference::InferenceEngine;
use crate::gguf::Loader;
use crate::tokenizer::TokenizerServiceBuilder;
use crate::session::SessionManager;
use candle_core::DType;
use clap::{Parser, Subcommand};
use std::{ffi::OsString, str::FromStr};
use colored::*;

type StreamCallback = Box<dyn FnMut(&str) -> anyhow::Result<()>>;

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
        #[arg(short = 'm', long = "max_tokens", default_value = "100")]
        max_tokens: usize,
        #[arg(short = 't', long = "temperature", default_value = "0.8")]
        temperature: f32,
        #[arg(short = 'd', long = "dtype", default_value = "f32")]
        compute_dtype: String,
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
        #[arg(short = 'm', long = "max_tokens", default_value = "100")]
        max_tokens: usize,
        #[arg(short = 't', long = "temperature", default_value = "0.8")]
        temperature: f32,
        #[arg(short = 'd', long = "dtype", default_value = "f32")]
        compute_dtype: String,
        #[arg(short = 'l', long = "max_seq_len", default_value_t = 32_000)]
        max_seq_len: usize,
        #[arg(short = 'f', long = "use_flash")]
        use_flash: bool,
        #[arg(short = 's', long = "stream")]
        stream: bool,
    }
    
}

pub fn run() -> anyhow::Result<()> {
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
            let tokenizer = TokenizerServiceBuilder::from_gguf_metadata(&info.kv_meta)?;
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
            compute_dtype,
            max_seq_len,
            use_flash,
            stream,
        } => {
            let path_str = model_path.to_string_lossy();
            let path_sanitized = path_str.replace('\\', "/");
            let dtype = DType::from_str(&compute_dtype).map_err(|_|
                anyhow::anyhow!("Invalid dtype: {}", compute_dtype))?;
            let mut inference_engine = InferenceEngine::builder()
                .model_path(&path_sanitized)
                .dtype(dtype)
                .max_seq_len(max_seq_len)
                .conv_on_cpu(true)
                .build()?;

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
                max_tokens,
                temperature as f64,
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
        Commands::Chat { prompt, model_path, system_message, max_tokens, temperature, compute_dtype, max_seq_len, use_flash, stream } => {
            let path_str = model_path.to_string_lossy();
            let path_sanitized = path_str.replace('\\', "/");
            let dtype = DType::from_str(&compute_dtype).map_err(|_|
                anyhow::anyhow!("Invalid dtype: {}", compute_dtype))?;
            let mut inference_engine = InferenceEngine::builder()
                .model_path(&path_sanitized)
                .dtype(dtype)
                .max_seq_len(max_seq_len)
                .conv_on_cpu(true)
                .build()?;

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
            let mut session = session_manager.start_session(&system_message);
            session.add_user_message(&prompt);
            let prompt_templated = session.get_messages();

            let report = inference_engine.generate_chat_stream(
                prompt_templated,
                max_tokens,
                temperature as f64,
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
    }

    anyhow::Ok(())
}
