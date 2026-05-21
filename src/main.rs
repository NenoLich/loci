mod cli;
mod config;
mod error;
mod gguf;
mod model;
mod tokenizer;
mod inference;
mod session;
mod api;

use tokio;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("debug"))
        )
        .with_writer(std::io::stderr)
        .init();
        
    cli::run().await
}
