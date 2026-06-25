use axum::{
    Router,
    routing::{get, post},
};

use tokio_util::sync::CancellationToken;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::info;

use crate::api::handlers::{AppState, chat_completions};
use crate::api::worker::WorkerCommand;

use tokio::sync::mpsc::Sender;

pub fn create_router(command_tx: Sender<WorkerCommand>) -> Router {
    let state = AppState { command_tx };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state)
}

pub async fn run_server(
    command_tx: Sender<WorkerCommand>,
    addr: &str,
    cancelation_token: CancellationToken,
) -> anyhow::Result<()> {
    let app = create_router(command_tx);
    let listener = tokio::net::TcpListener::bind(addr).await?;

    let shutdown_trigger = cancelation_token.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to listen for ctrl_c");
        shutdown_trigger.cancel();
    });
    let axum_shutdown_token = cancelation_token.clone();

    info!("🦀 Loci API server listening on http://{}", addr);
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            axum_shutdown_token.cancelled().await;
        })
        .await?;

    info!("Server stopped. Cleaning up resources...");
    Ok(())
}
