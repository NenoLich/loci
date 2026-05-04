mod cli;
mod config;
mod error;
mod gguf;
mod model;
mod tokenizer;
mod inference;
mod session;

use tracing_subscriber::EnvFilter;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("debug"))
        )
        .with_writer(std::io::stderr)
        .init();
        
    cli::run()
}
