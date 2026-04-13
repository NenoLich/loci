mod loader;
mod gguf_types;
mod tokenizer;
mod model;

use std::fs::File;
use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Instant;
use candle_core::{Device, Tensor};
use clap::{Parser, Subcommand};
use memmap2::MmapOptions;
use anyhow::Context;
use crate::tokenizer::LlmTokenizer;
use crate::model::LlmModel;
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
    Info { 
        path: OsString,
        
        /// Count of tensors to show
        #[arg(name = "first_k_tensors", short = 'k', long = "first_k", default_value_t = 10)]
        first_k_tensors: usize,
     },

    /// Tokenize text
    Tokenize { text: String },

    Forward { text: String },
}

fn main() -> anyhow::Result<()> {
    let args = Cli::parse();

    match args.command {
        Commands::Info { path, first_k_tensors} => {
            let path_str = path.to_string_lossy();
            let sanitized = path_str.replace('\\', "/");
            let path_buf = PathBuf::from(sanitized);
            let loader = Loader;
            let _info = loader.load_gguf_info(&path_buf, first_k_tensors, true)?;
        }, 
        Commands::Tokenize { text } => {
            let loader = Loader;
            let info = loader.load_gguf_info("models/LFM2.5-350M-F16.gguf", 10, false)?;
            let tokenizer = LlmTokenizer::from_gguf_metadata(&info.kv_meta)?;
            let encoding = tokenizer.encode(&text)?;

            println!("Input: \"{}\"", text);
            let tokens = encoding.get_ids();
            println!("Tokens: {:?}", tokens);
            let decoded = tokenizer.decode(encoding.get_ids())?;
            println!("Decoded: \"{}\"", decoded.trim());
        },
        Commands::Forward { text } => {
            let file_path = "models/LFM2.5-350M-F16.gguf";
            let loader = Loader;
            let info = loader.load_gguf_info(file_path, 10, false)?;
            let tokenizer = LlmTokenizer::from_gguf_metadata(&info.kv_meta)?;
            let encoding = tokenizer.encode(&text)?;

            println!("Input: \"{}\"", text);
            let tokens = encoding.get_ids();
            println!("Tokens: {:?}", tokens);

            let start_embd_time = Instant::now();
            let embedding_tensor_info = info.tensor_info.iter()
                .find(|&entry| entry.name.contains("token_embd.weight"))
                .context("Could not find 'token_embd.weight' in GGUF file")?;

            let model = LlmModel;
            let file = File::open(file_path)?;
            let mmap = unsafe {
                MmapOptions::new().map(&file)?
            };
            let tensor_offset_start = info.tensor_offset_start;
            let embedding_tensor_raw = model.load_tensor(&mmap, embedding_tensor_info, tensor_offset_start)?;

            let tensor_token = Tensor::from_slice(tokens, tokens.len(), &Device::cuda_if_available(0)?)?;
            let embeddings_narrow = embedding_tensor_raw.index_select(&tensor_token, 1)?;

            let embeddings = embeddings_narrow.t()?.contiguous()?;
            println!("Embeddings shape: {:?}", embeddings.shape());
            let first_10_emb = embeddings.get(0)?
                .narrow(0, 0, 10.min(embeddings.dim(1)?))?
                .to_dtype(candle_core::DType::F32)?
                .to_vec1::<f32>()?;
            println!("Embedding completed in {:.2}s", start_embd_time.elapsed().as_secs_f64());
            println!("First token embedding (first 10 values): {:?}", &first_10_emb);
        },
    }

    anyhow::Ok(())
}
