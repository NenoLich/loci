use tokio::time::{sleep, Duration, Instant};
use tokio::sync::mpsc::{Receiver, Sender, channel};
use tokio::sync::oneshot;
use tokio::sync::mpsc::error::TrySendError::{Closed, Full};
use tokio_util::sync::CancellationToken;
use axum::response::sse::Event;
use axum::http::StatusCode;
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};
use std::convert::Infallible;
use std::sync::Arc;
use tracing::{info, error, debug};
use uuid::Uuid;

use crate::inference::{InferenceEngine, StreamCallback, StreamFrame, GenerationDataType, GenerationReport};
use crate::error::LociError;
use crate::gguf::GgufInfo;
use crate::config::{InferenceConfig, GenerationOverrides, ModelCacheConfig};
use crate::api::types::*;

#[derive(Clone)]
pub struct StaticChatCompletionData {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
}

pub enum WorkerCommand {
    ChatCompletion {
        req: ValidatedChatCompletionRequest,
        // We use a oneshot channel to send the SSE stream back to the handler
        response_tx: oneshot::Sender<Result<Receiver<Result<Event, Infallible>>, String>>,
    },
}

pub struct EngineWorker {
    inference_config: InferenceConfig,
    model_cache_config: ModelCacheConfig,
    active_engine: Option<InferenceEngine>,
    command_rx: Receiver<WorkerCommand>,
    idle_timeout: Duration,
    cancellation_token: CancellationToken,
}

impl EngineWorker {
    pub fn new(inference_config: InferenceConfig, model_cache_config: ModelCacheConfig, active_engine: Option<InferenceEngine>, command_rx: Receiver<WorkerCommand>, idle_timeout: u64, cancellation_token: CancellationToken) -> Self {
        EngineWorker {
            inference_config,
            model_cache_config,
            active_engine,
            command_rx,
            idle_timeout: Duration::from_secs(idle_timeout),
            cancellation_token,
        }
    }

    pub async fn run(mut self) {
        let mut last_used = Instant::now();
        
        loop {
            // We use tokio::select to watch for incoming commands AND the idle timeout at the same time
            tokio::select! {               
                // Scenario 1: A new HTTP request arrived
                maybe_cmd = self.command_rx.recv() => {
                    match maybe_cmd {
                        Some(WorkerCommand::ChatCompletion { req, response_tx }) => {
                            // 1. Check/Swap the model safely
                            let current_model_match = self.active_engine.as_ref()
                                .map(|s| s.model_path() == req.model)
                                .unwrap_or(false);

                            if !current_model_match {
                                // Load new model (automatically drops old one)
                                self.drop_engine();
                                let builder = InferenceEngine::builder()
                                    .with_gguf_metadata(req.model.clone())
                                    .inference_config(Some(self.inference_config.clone()))
                                    .model_cache_config(Some(self.model_cache_config.clone()));

                                let engine_result = tokio::task::spawn_blocking(move || {
                                    builder.build()
                                }).await.unwrap();

                                match engine_result {
                                    Ok(eng) => {
                                        self.active_engine = Some(eng);
                                    }
                                    Err(e) => {
                                        let _ = response_tx.send(Err(e.to_string()));
                                        continue;
                                    }
                                }
                            }
                            
                            let (stream_tx, stream_rx) = tokio::sync::mpsc::channel(32);

                            // Send the receiving end back to the Axum handler immediately
                            let _ = response_tx.send(Ok(stream_rx));

                            if let Some(mut engine) = self.active_engine.take() {
                                let generation_cancel_token = self.cancellation_token.clone();
                                // 3. Process the inference synchronously inside this blocking thread.
                                let returned_engine = tokio::task::spawn_blocking(move || {
                                    if req.stream.unwrap_or(false) {
                                        run_stream_generation(&mut engine, req, stream_tx, generation_cancel_token);
                                    } else {
                                        // For non-streaming, run generation and send a single event
                                        run_single_generation(&mut engine, req, stream_tx, generation_cancel_token);
                                    }
                                    engine
                                }).await.unwrap();

                                self.active_engine = Some(returned_engine);
                            }

                            // Update tracking timestamp
                            last_used = Instant::now();
                        }
                        None => {
                        // Channel closed! Server is shutting down.
                        info!("Worker channel closed. Shutting down worker gracefully...");
                        break;
                        }
                    }
                }

                // Scenario 2: No requests came in, check if we should unload the model
                _ = sleep(Duration::from_secs(30)) => {
                    if self.active_engine.is_some() && last_used.elapsed() > self.idle_timeout {
                        info!("Engine idle timeout reached. Unloading model.");
                        self.drop_engine(); // Drops the engine and frees VRAM/RAM
                    }
                }
            }
        }

        self.drop_engine();
    }

    fn drop_engine(&mut self) {
        if let Some(engine) = self.active_engine.as_mut() {
            engine.flush_cache_to_file();
        }
        self.active_engine = None;
    }
}


fn run_stream_generation(
    engine: &mut InferenceEngine,
    req: ValidatedChatCompletionRequest,
    stream_tx: Sender<Result<Event, Infallible>>,
    cancellation_token: CancellationToken,
) -> () {
    let overrides = GenerationOverrides::new(
        req.temperature,
        req.top_p,
        req.max_tokens,
        req.repetition_penalty,
        Some(req.tool_choice),
        req.reasoning_effort,
        req.stop,
        req.logprobs,
        req.top_logprobs,
        req.seed,
        None,
    );

    // Setup initial chunk
    let static_data = StaticChatCompletionData {
        id: Uuid::new_v4().to_string(),
        created: std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: req.model.clone(),
        system_fingerprint: format!("loci-{}", req.model),
    };

    let initial_chunk = build_initial_chunk(static_data.clone());
    if let Ok(event) = Event::default().json_data(initial_chunk) {
        if stream_tx.try_send(Ok(event)).is_err() {
            // The client disconnected (closed browser tab)
            info!("User disconnected. Stopping generation.");
            return;
        }
    };
    
    let callback_tx = stream_tx.clone();
    let callback_static_data = static_data.clone();
    let callback: StreamCallback = 
        Box::new(move |frame_data| {
            if cancellation_token.is_cancelled() {
                info!("Server shutdown signal received. Aborting generation immediately.");
                return Err(LociError::Stream("Server shutdown signal received.".to_string()));
            }
            let regular_chunk = build_regular_chunk(callback_static_data.clone(), frame_data);
            let event = Event::default().json_data(regular_chunk)
                .map_err(|e| LociError::Stream(e.to_string()))?;
            match callback_tx.try_send(Ok(event)) {
                Err(Full(message)) => {
                    std::thread::sleep(std::time::Duration::from_micros(10));
                    // Try one more time after the sleep
                    if callback_tx.try_send(message).is_err() {
                        return Err(LociError::Stream("Stream channel backed up or closed".to_string()));
                    };
                }
                Err(Closed(_)) => {
                    // The client disconnected (closed browser tab)
                    info!("User disconnected. Stopping generation.");
                    return Err(LociError::Stream("User disconnected".to_string()));
                }
                _ => (),
            }

            Ok(())
        });
    match engine.generate_chat_stream(&req.messages, &req.tools.unwrap_or_default(), overrides, callback) {
        Ok(report) => {
            debug!("Generation report: {:#?}", report);
            let final_content_chunk = build_final_content_chunk(static_data.clone(), report.finish_reason.clone());
            if let Ok(final_chunk_event) = Event::default().json_data(final_content_chunk) {
                if stream_tx.try_send(Ok(final_chunk_event)).is_err() {
                    // The client disconnected (closed browser tab)
                    info!("User disconnected. Stopping generation.");
                    return;
                }
            }
            if matches!(req.stream_options, Some(options) if options.include_usage == Some(true)) {
                let usage_chunk = build_usage_chunk(static_data.clone(), report);
                if let Ok(usage_chunk_event) = Event::default().json_data(usage_chunk) {
                    if stream_tx.try_send(Ok(usage_chunk_event)).is_err() {
                        // The client disconnected (closed browser tab)
                        info!("User disconnected. Stopping generation.");
                        return;
                    }
                }
            }

        },
        Err(e) => {
            if e.to_string().contains("User disconnected") {
                info!("Generation stopped early: client closed the connection.");
            } else {
                let error_msg = format!("Generation failed with an engine error: {}", e);
                error!("{}", &error_msg);
                let json_error = json!({
                    "error": {
                        "message": error_msg,
                        "type": "engine_error",
                        "code": 500
                    }
                });
                if let Ok(event) = Event::default().json_data(json_error) {
                    if stream_tx.try_send(Ok(event)).is_err() {
                        // The client disconnected (closed browser tab)
                        info!("User disconnected. Stopping generation.");
                        return;
                    }
                };
            }
        }
    }
}

fn build_initial_chunk(
    static_data: StaticChatCompletionData,
) -> ChatCompletionChunk {
    let choices = vec![
        ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: Some(Role::Assistant),
                content: Some(String::from("")),
                reasoning_content: None,
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: None,
        }
    ];
    ChatCompletionChunk {
        id: static_data.id,
        object: "chat.completion.chunk".to_string(),
        created: static_data.created,
        model: static_data.model,
        system_fingerprint: static_data.system_fingerprint,
        choices,
        usage: None,
    }
}

fn build_regular_chunk<'a>(
    static_data: StaticChatCompletionData,
    frame_data: StreamFrame<'a>,
) -> ChatCompletionChunk {
    let role = None;
    let delta = match frame_data.output_type {
        GenerationDataType::DirectContent => ChunkDelta {
            role,
            content: Some(frame_data.output.to_string()),
            reasoning_content: None,
            tool_calls: None,
        },
        GenerationDataType::ToolCallName | GenerationDataType::ToolCallArguments => ChunkDelta {
            role,
            content: None,
            reasoning_content: None,
            tool_calls: frame_data.tool_call_chunk
                            .map(|tc| vec![tc])
        },
        GenerationDataType::Reasoning => ChunkDelta {
            role,
            content: None,
            reasoning_content: Some(frame_data.output.to_string()),
            tool_calls: None,
        },
    };

    ChatCompletionChunk {
        id: static_data.id,
        object: "chat.completion.chunk".to_string(),
        created: static_data.created,
        model: static_data.model,
        system_fingerprint: static_data.system_fingerprint,
        choices: vec![
            ChunkChoice {
                index: 0,
                delta,
                logprobs: frame_data.logprobs
                    .map(|cl| ChunkLogprob {
                        content:vec![cl]
                    }),
                finish_reason: None,
            }
        ],
        usage: None,
    }
}

fn build_final_content_chunk(
    static_data: StaticChatCompletionData,
    finish_reason: FinishReason,
) -> ChatCompletionChunk {
    let choice = ChunkChoice {
        index: 0,
        delta: ChunkDelta {
            role: None,
            content: None,
            tool_calls: None,
            reasoning_content: None,
        },
        logprobs: None,
        finish_reason: Some(finish_reason),
    };

    ChatCompletionChunk {
        id: static_data.id,
        object: "chat.completion.chunk".to_string(),
        created: static_data.created,
        model: static_data.model,
        system_fingerprint: static_data.system_fingerprint,
        choices: vec![choice],
        usage: None,
    }
}

fn build_usage_chunk(
    static_data: StaticChatCompletionData,
    report: GenerationReport,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: static_data.id,
        object: "chat.completion.chunk".to_string(),
        created: static_data.created,
        model: static_data.model,
        system_fingerprint: static_data.system_fingerprint,
        choices: vec![],
        usage: Some(report.usage),
    }
}

fn run_single_generation(
    engine: &mut InferenceEngine,
    req: ValidatedChatCompletionRequest,
    stream_tx: Sender<Result<Event, Infallible>>,
    cancellation_token: CancellationToken,
) -> () {
    let overrides = GenerationOverrides::new(
        req.temperature,
        req.top_p,
        req.max_tokens,
        req.repetition_penalty,
        Some(req.tool_choice),
        req.reasoning_effort,
        req.stop,
        req.logprobs,
        req.top_logprobs,
        req.seed,
        None,
    );

    let callback_tx = stream_tx.clone();
    let callback: StreamCallback = Box::new(move |_| {
        if callback_tx.is_closed() {
                info!("User disconnected. Stopping generation.");
                return Err(LociError::Stream("User disconnected".to_string()));
            }
        if cancellation_token.is_cancelled() {  
            info!("Server shutdown signal received. Aborting generation immediately.");
            return Err(LociError::Stream("Server shutdown signal received.".to_string()));
        }
        Ok(())
    });
    match engine.generate_chat_stream(&req.messages, &req.tools.unwrap_or_default(), overrides, callback) {
        Ok(report) => {
            debug!("Generation report: {:#?}", report);
            let chat_completion_response = build_chat_completion_response(&req.model, report);
            if let Ok(response_event) = Event::default().json_data(chat_completion_response) {
                match stream_tx.try_send(Ok(response_event)) {
                    Err(Full(event)) => {
                        std::thread::sleep(std::time::Duration::from_micros(10));
                        // Try one more time after the sleep
                        if stream_tx.try_send(event).is_err() {
                            info!("Stream channel backed up or closed");
                        };
                    }
                    Err(Closed(_)) => {
                        // The client disconnected (closed browser tab)
                        info!("User disconnected. Stopping generation.");
                        return;;
                    }
                    _ => (),
                };
            }
        },
        Err(e) => {
            if e.to_string().contains("User disconnected") {
                info!("Generation stopped early: client closed the connection.");
            } else {
                let error_msg = format!("Generation failed with an engine error: {}", e);
                error!("{}", &error_msg);
                let json_error = json!({
                    "error": {
                        "message": error_msg,
                        "type": "engine_error",
                        "code": 500
                    }
                });
                if let Ok(event) = Event::default().json_data(json_error) {
                    if stream_tx.try_send(Ok(event)).is_err() {
                        // The client disconnected (closed browser tab)
                        info!("User disconnected. Stopping generation.");
                        return;
                    }
                };
            }
        }
    }
}

fn build_chat_completion_response(
    model: &str,
    report: GenerationReport,
) -> ChatCompletionResponse {
        let choices = vec![Choice {
            index: 0,
            message: report.chat_message,
            finish_reason: Some(report.finish_reason),
        }];
    ChatCompletionResponse {
        id: Uuid::new_v4().to_string(),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: model.to_string(),
        system_fingerprint: format!("loci-{}", model),
        choices,
        usage: Some(report.usage),
    }
}
