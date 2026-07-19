use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let fmt_layer = fmt::layer().with_writer(std::io::stderr);

    #[cfg(feature = "profiling")]
    let (flame_layer, _guard) = {
        let enable_flame = std::env::var("LOCI_FLAMEGRAPH")
            .map(|v| v == "true")
            .unwrap_or(false);
        if enable_flame {
            let flame_path = std::env::var("LOCI_FLAMEGRAPH_PATH")
                .unwrap_or_else(|_| ".profile/tracing.folded".to_string());
            let (flame_layer, _guard) = tracing_flame::FlameLayer::with_file(&flame_path)?;
            (Some(flame_layer), Some(_guard))
        } else {
            (None, None)
        }
    };
    #[cfg(not(feature = "profiling"))]
    let (flame_layer, _guard): (Option<tracing_subscriber::layer::Identity>, Option<()>) =
        (None, None);

    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug")))
        .with(fmt_layer)
        .with(flame_layer)
        .init();

    loci::cli::run().await
}
