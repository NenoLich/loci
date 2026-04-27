mod gguf_types;
mod inference;
mod loader;
mod model;
mod model_config;
mod tokenizer;

use crate::inference::InferenceEngine;
use crate::loader::Loader;
use crate::tokenizer::TokenizerServiceBuilder;
use candle_core::DType;
use clap::{Parser, Subcommand};
use std::{ffi::OsString, str::FromStr};
use colored::*;

#[derive(Parser)]
#[command(name = "loci")]
#[command(about = "Local LLM inference tool")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
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
    Tokenize { text: String },

    /// Generate text from a prompt
    Generate {
        prompt: String,
        #[arg(short = 'm', long = "max_tokens", default_value = "100")]
        max_tokens: usize,
        #[arg(short = 't', long = "temperature", default_value = "0.8")]
        temperature: f32,
        #[arg(short = 'd', long = "dtype", default_value = "f32")]
        compute_dtype: String,
        #[arg(short = 'f', long = "use_flash")]
        use_flash: bool,
        #[arg(short = 's', long = "stream")]
        stream: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let args = Cli::parse();

    match args.command {
        Commands::Info {
            path,
            first_k_tensors,
        } => {
            let path_str = path.to_string_lossy();
            let path_sanitized = path_str.replace('\\', "/");
            let loader = Loader;
            let _info = loader.load_gguf_info(&path_sanitized, first_k_tensors, true)?;
        }
        Commands::Tokenize { text } => {
            let loader = Loader;
            let info = loader.load_gguf_info("models/LFM2.5-350M-F16.gguf", 10, false)?;
            let tokenizer = TokenizerServiceBuilder::from_gguf_metadata(&info.kv_meta)?;
            let encoding = tokenizer.encode(&text)?;
            
            println!("Input: \"{}\"", text);
            let tokens = encoding.get_ids();
            println!("Tokens: {:?}", tokens);
            let decoded = tokenizer.decode(encoding.get_ids())?;
            println!("Decoded: \"{}\"", decoded.trim());
        }
        Commands::Generate {
            prompt,
            max_tokens,
            temperature,
            compute_dtype,
            use_flash,
            stream,
        } => {
            let loader = Loader;
            let info = loader.load_gguf_info("models/LFM2.5-350M-F16.gguf", 10, false)?;
            let dtype = DType::from_str(&compute_dtype).map_err(|_| 
                anyhow::anyhow!("Invalid dtype: {}", compute_dtype))?;
            let mut inferenence_engine = InferenceEngine::new(&info, dtype, use_flash)?;
            
            println!("🦀 Generating: \"{}\"", prompt);
            let start = std::time::Instant::now();

            if stream {
                inferenence_engine.generate_stream(
                    prompt.as_str(), 
                    max_tokens, 
                    temperature as f64, 
                    |output_chunk| {
                        print!("{}", output_chunk.green());
                        use std::io::Write;
                        std::io::stdout().flush()?;
                        anyhow::Ok(())
                    }
                )?;
            } else {
                let output = inferenence_engine.generate(prompt.as_str(), max_tokens, temperature as f64)?;
 
                println!("\n✨ Output:\n{}", output);
            }
            
            let elapsed = start.elapsed();
            println!(
                "\n⏱️  Generated {} tokens in {:.2}s ({:.2} tok/s)",
                max_tokens,
                elapsed.as_secs_f64(),
                max_tokens as f64 / elapsed.as_secs_f64()
            );
        }
    }

    anyhow::Ok(())
}
