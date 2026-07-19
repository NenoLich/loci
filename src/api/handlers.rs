use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, sse::Sse},
};
use serde_json::json;
use tokio::sync::oneshot;

use crate::{api::types::*, api::worker::WorkerCommand};

#[derive(Clone)]
pub struct AppState {
    pub command_tx: tokio::sync::mpsc::Sender<WorkerCommand>,
}

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<AppState>,
    req: ValidatedChatCompletionRequest,
) -> Response {
    let (response_tx, response_rx) = oneshot::channel();

    // 1. Send the request to our single-threaded background worker
    if state
        .command_tx
        .send(WorkerCommand::ChatCompletion { req, response_tx })
        .await
        .is_err()
    {
        return (StatusCode::INTERNAL_SERVER_ERROR, "Worker thread died").into_response();
    }

    // 2. Wait for the worker to initialize the model and give us our stream channel
    match response_rx.await {
        Ok(Ok(stream_rx)) => {
            // Turn the Mpsc Receiver into an Axum SSE Stream
            let stream = tokio_stream::wrappers::ReceiverStream::new(stream_rx);
            Sse::new(stream).into_response()
        }
        Ok(Err(server_error)) => {
            let error =
                json!({"error": {"message": server_error, "type": "server_error", "code": 500}});
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response()
        }
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Worker dropped request").into_response(),
    }
}
