mod loader;
mod gguf_types;
mod tokenizer;

use std::path::PathBuf;
use clap::{Parser, Subcommand};
use crate::tokenizer::LlmTokenizer;

use crate::loader::Loader;

#[derive(Parser)]
#[command(name="loci")]
#[command(about="Local LLM inference tool")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Info {path: PathBuf},

    /// Tokenize text
    Tokenize { text: String },
}

fn main() -> anyhow::Result<()> {
    let args = Cli::parse();

    match args.command {
        Commands::Info { path } => {
            let loader = Loader;
            let _info = loader.load_gguf_info(&path, true)?;
        }, 
        Commands::Tokenize { text } => {
            let loader = Loader;
            let info = loader.load_gguf_info("models/LFM2.5-350M-F16.gguf", false)?;
            let tokenizer = LlmTokenizer::from_gguf_metadata(&info.kv_meta)?;
            let encoding = tokenizer.encode(&text)?;

            println!("Input: \"{}\"", text);
            let tokens = encoding.get_ids();
            println!("Tokens: {:?}", tokens);
            let decoded = tokenizer.decode(encoding.get_ids())?;
            println!("Decoded: \"{}\"", decoded.trim());
        },
    }

    anyhow::Ok(())
}
