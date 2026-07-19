mod fixtures;

use axum::http::StatusCode;
use axum::response::sse::Event;
use axum_test::{TestResponse, TestServer};
use loci::api::server::create_router;
use loci::api::worker::WorkerCommand;
use serde_json::json;
use tokio::sync::mpsc;

#[tokio::test]
async fn test_health() {
    let (command_tx, _command_rx) = mpsc::channel(32);
    let server = TestServer::new(create_router(command_tx));
    let response: TestResponse = server.get("/health").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    assert_eq!(response.text(), "OK");
}

#[tokio::test]
async fn test_chat_completion_requires_model_field() {
    let (command_tx, _command_rx) = mpsc::channel(32);
    let server = TestServer::new(create_router(command_tx));

    let response: TestResponse = server
        .post("/v1/chat/completions")
        .json(&json!({
            "messages": [{"role": "user", "content": "Hello"}]
        }))
        .await;

    assert_eq!(response.status_code(), StatusCode::BAD_REQUEST);
    let body: serde_json::Value = response.json();
    assert_eq!(body["error"]["message"], "model is required");
}

#[tokio::test]
async fn test_chat_completion_worker_returns_error() {
    let (command_tx, mut command_rx) = mpsc::channel(32);

    tokio::spawn(async move {
        if let Some(WorkerCommand::ChatCompletion {
            req: _,
            response_tx,
        }) = command_rx.recv().await
        {
            let _ = response_tx.send(Err("Model not found".to_string()));
        }
    });

    let server = TestServer::new(create_router(command_tx));

    let response: TestResponse = server
        .post("/v1/chat/completions")
        .json(&json!({
            "model": "nonexistent.gguf",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        }))
        .await;

    assert_eq!(response.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
    let body: serde_json::Value = response.json();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Model not found")
    );
}

#[tokio::test]
async fn test_chat_completion_worker_drops_request() {
    let (command_tx, mut command_rx) = mpsc::channel(32);

    tokio::spawn(async move {
        if let Some(WorkerCommand::ChatCompletion { response_tx, .. }) = command_rx.recv().await {
            drop(response_tx);
        }
    });

    let server = TestServer::new(create_router(command_tx));

    let response: TestResponse = server
        .post("/v1/chat/completions")
        .json(&json!({
            "model": "test.gguf",
            "messages": [{"role": "user", "content": "Hello"}],
        }))
        .await;

    assert_eq!(response.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(response.text(), "Worker dropped request");
}

#[tokio::test]
async fn test_chat_completion_streaming_sse() {
    let (command_tx, mut command_rx) = mpsc::channel(32);

    tokio::spawn(async move {
        while let Some(cmd) = command_rx.recv().await {
            match cmd {
                WorkerCommand::ChatCompletion {
                    req: _,
                    response_tx,
                } => {
                    let (stream_tx, stream_rx) = mpsc::channel(128);

                    let initial = json!({
                        "id": "test-id",
                        "object": "chat.completion.chunk",
                        "created": 12345,
                        "model": "test-model",
                        "system_fingerprint": "loci-test-model",
                        "choices": [{
                            "index": 0,
                            "delta": { "role": "assistant", "content": "" },
                            "logprobs": null,
                            "finish_reason": null
                        }]
                    });
                    stream_tx
                        .send(Ok(Event::default().json_data(initial).unwrap()))
                        .await
                        .unwrap();

                    let content = json!({
                        "id": "test-id",
                        "object": "chat.completion.chunk",
                        "created": 12345,
                        "model": "test-model",
                        "system_fingerprint": "loci-test-model",
                        "choices": [{
                            "index": 0,
                            "delta": { "content": "Hello" },
                            "logprobs": null,
                            "finish_reason": null
                        }]
                    });
                    stream_tx
                        .send(Ok(Event::default().json_data(content).unwrap()))
                        .await
                        .unwrap();

                    let final_chunk = json!({
                        "id": "test-id",
                        "object": "chat.completion.chunk",
                        "created": 12345,
                        "model": "test-model",
                        "system_fingerprint": "loci-test-model",
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "logprobs": null,
                            "finish_reason": "stop"
                        }]
                    });
                    stream_tx
                        .send(Ok(Event::default().json_data(final_chunk).unwrap()))
                        .await
                        .unwrap();

                    let _ = response_tx.send(Ok(stream_rx));
                }
            }
        }
    });

    let server = TestServer::new(create_router(command_tx));

    let response: TestResponse = server
        .post("/v1/chat/completions")
        .json(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        }))
        .await;

    assert_eq!(response.status_code(), StatusCode::OK);

    let body = response.text();
    assert!(body.contains("test-id"));
    assert!(body.contains("chat.completion.chunk"));
    assert!(body.contains("Hello"));
    assert!(body.contains("stop"));
}
