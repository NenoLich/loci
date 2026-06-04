use axum::{
    routing::{get, post},
    Router,
};
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::info;

use crate::config::InferenceConfig;
use crate::inference::InferenceEngine;
use std::time::Instant;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;
use crate::api::handlers::{chat_completions, AppState};
use crate::api::worker::WorkerCommand;

pub fn create_router(command_tx: Sender<WorkerCommand>) -> Router {
    let state = AppState { 
        command_tx,
     };
    
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

pub async fn run_server(command_tx: Sender<WorkerCommand>, addr: &str) -> anyhow::Result<()> {
    let app = create_router(command_tx);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    
    info!("🦀 Loci API server listening on http://{}", addr);
    
    axum::serve(listener, app).await?;
    Ok(())
}