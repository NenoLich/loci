mod api;
mod cli;
mod config;
mod error;
mod gguf;
mod inference;
mod model;
mod render;
mod session;
mod tokenizer;
mod types;

use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let fmt_layer = fmt::layer().with_writer(std::io::stderr);

    let enable_flame = std::env::var("LOCI_FLAMEGRAPH")
        .map(|v| v == "true")
        .unwrap_or(false);
    let (flame_layer, _guard) = if enable_flame {
        let flame_path = std::env::var("LOCI_FLAMEGRAPH_PATH")
            .unwrap_or_else(|_| ".profile/tracing.folded".to_string());
        let (flame_layer, _guard) = tracing_flame::FlameLayer::with_file(&flame_path)?;
        (Some(flame_layer), Some(_guard))
    } else {
        (None, None)
    };

    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug")))
        .with(fmt_layer)
        .with(flame_layer)
        .init();

    cli::run().await
}
