use tokio::time::{sleep, Duration, Instant};
use tokio::sync::mpsc::{Receiver, Sender, channel};
use tokio::sync::oneshot;
use axum::response::sse::Event;
use axum::http::StatusCode;
use serde_json::json;
use std::time::SystemTime;
use std::convert::Infallible;
use std::sync::Arc;
use tracing::{info, error};

use crate::inference::{InferenceEngine, StreamCallback};
use crate::config::GenerationOverrides;
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
    config: InferenceConfig,
    active_engine: Option<Arc<InferenceEngine>>,
    command_rx: Receiver<WorkerCommand>,
    idle_timeout: Duration,
}

impl EngineWorker {
    pub fn new(config: InferenceConfig, active_engine: Option<InferenceEngine>, command_rx: Receiver<WorkerCommand>, idle_timeout: u64) -> Self {
        EngineWorker {
            config,
            active_engine: active_engine.map(Arc::new),
            command_rx,
            idle_timeout: Duration::from_secs(idle_timeout),
        }
    }

    pub async fn run(mut self) {
        let mut last_used = Instant::now();

        loop {
            // We use tokio::select to watch for incoming commands AND the idle timeout at the same time
            tokio::select! {
                // Scenario 1: A new HTTP request arrived
                Some(cmd) = self.command_rx.recv() => {
                    match cmd {
                        WorkerCommand::ChatCompletion { req, response_tx } => {
                            // 1. Check/Swap the model safely
                            let current_model_match = self.active_engine.as_ref()
                                .map(|s| s.model_path() == req.model)
                                .unwrap_or(false);

                            if !current_model_match {
                                // Load new model (automatically drops old one)
                                match InferenceEngine::builder()
                                    .with_gguf_metadata(req.model.clone())
                                    .config(self.config)
                                    .build() 
                                {
                                    Ok(eng) => {
                                        self.active_engine = Some(Arc::new(eng));
                                    }
                                    Err(e) => {
                                        let _ = response_tx.send(Err(e.to_string()));
                                        continue;
                                    }
                                }
                            }

                            // 2. We have the engine! Setup your channel streaming
                            let engine = self.active_engine.unwrap().clone();
                            let (stream_tx, stream_rx) = tokio::sync::mpsc::channel(32);

                            // Send the receiving end back to the Axum handler immediately
                            let _ = response_tx.send(Ok(stream_rx));

                            // 3. Process the inference synchronously inside this worker thread.
                            // This naturally forces requests into a single-file line!
                            if req.stream.unwrap_or(false) {
                                // For streaming, we run your generator loop until it finishes
                                // filling the `tx` channel.
                                run_stream_generation(engine, req, stream_tx).await;
                            } else {
                                // For non-streaming, run generation and send a single event
                                run_single_generation(engine, req, stream_tx).await;
                            }

                            // Update tracking timestamp
                            last_used = Instant::now();
                        }
                    }
                }

                // Scenario 2: No requests came in, check if we should unload the model
                _ = sleep(Duration::from_secs(30)) => {
                    if self.active_engine.is_some() && last_used.elapsed() > self.idle_timeout {
                        info!("Engine idle timeout reached. Unloading model.");
                        self.active_engine = None; // Drops the engine and frees VRAM/RAM
                    }
                }
            }
        }
    }
}

async fn run_stream_generation(
    engine: Arc<InferenceEngine>,
    req: ValidatedChatCompletionRequest,
    stream_tx: Sender<Result<Event, Infallible>>,
) -> () {
    let overrides = GenerationOverrides::new(
        req.temperature,
        req.top_p,
        req.max_tokens,
        req.repetition_penalty,
        req.seed,
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

    let initial_chunk = build_initial_chunk(&static_data);
    if let Ok(event) = Event::default().json_data(initial_chunk) {
        let _ = stream_tx.send(Ok(event)).await;
    };
    
    let callback_tx = stream_tx.clone();
    let callback_static_data = static_data.clone();
    let callback: StreamCallback = 
        Box::new(move |chunk| {
            let event = Event::default()
            if callback_tx.blocking_send(Ok(event)).is_err() {
                anyhow::bail!("User disconnected");
            };
        });
    match engine.generate_chat_stream(&req.messages, &req.tools, overrides, true, callback) {
        Ok(report) => {
            
        },
        Err(e) => {
            if e.to_string().contains("User disconnected") {
                info!("Generation stopped early: client closed the connection.");
            } else {
                let error_msg = format!("Generation failed with an engine error: {}", e);
                error!(&error_msg);
                let json_error = json!({
                    "error": {
                        "message": error_msg,
                        "type": "engine_error",
                        "code": 500
                    }
                });
                if let Ok(event) = Event::default().json_data(json_error) {
                    let _ = stream_tx.send(Ok(event)).await;
                };
            }
        }
    }
}

fn build_initial_chunk(
    static_data: &StaticChatCompletionData,
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

fn build_content_chunk(
    static_data: &StaticChatCompletionData,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: static_data.id,
        object: "chat.completion.chunk".to_string(),
        created: static_data.created,
        model: static_data.model,
        system_fingerprint: static_data.system_fingerprint,
        choices: vec![],
        usage: None,
    }
}

fn build_final_content_chunk(
    static_data: &StaticChatCompletionData,
) -> ChatCompletionChunk {
    let choices = ChunkChoice {
        index: 0,
        delta: ChunkDelta {
            role: None,
            content: None,
            tool_calls: None,
            reasoning_content: None,
        },
        logprobs: None,
        finish_reason: None,
    }

    ChatCompletionChunk {
        id: static_data.id,
        object: "chat.completion.chunk".to_string(),
        created: static_data.created,
        model: static_data.model,
        system_fingerprint: static_data.system_fingerprint,
        choices: vec![],
        usage: None,
    }
}

fn build_usage_chunk(
    static_data: &StaticChatCompletionData,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: static_data.id,
        object: "chat.completion.chunk".to_string(),
        created: static_data.created,
        model: static_data.model,
        system_fingerprint: static_data.system_fingerprint,
        choices: vec![],
        usage: None,
    }
}

async fn generate_chat_completion(
    engine: Arc<InferenceEngine>,
    req: ValidatedChatCompletionRequest,
) -> anyhow::Result<GenerationReport> {
    let prompt = build_prompt_from_messages(&req.messages);
    let temperature = req.temperature.unwrap_or(0.7) as f64;
    let max_tokens = req.max_tokens.unwrap_or(256);
    
    // Call your existing inference engine
    state.engine.generate_chat_stream(
        &prompt,
        max_tokens,
        temperature,
        false,  // use_flash
        |_| Ok(()),  // no-op callback for non-streaming
    )
}

fn stream_chat_completion(
    engine: Arc<InferenceEngine>,
    req: ValidatedChatCompletionRequest,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel(32);
    
    // Spawn generation in background
    tokio::spawn(async move {
        let prompt = build_prompt_from_messages(&req.messages);
        let temperature = req.temperature.unwrap_or(0.7) as f64;
        let max_tokens = req.max_tokens.unwrap_or(256);
        let model_name = req.model.unwrap_or_else(|| "loci-local".to_string());
        let id = Uuid::new_v4().to_string();
        let created = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Callback: send each token chunk via SSE
        let callback = move |chunk: &str| {
            let event = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: Some(chunk.to_string()),
                    },
                    finish_reason: None,
                }],
            };
            let sse_event = Event::default().json_data(event).unwrap();
            tx.blocking_send(Ok(sse_event))
                .map_err(|_| anyhow::anyhow!("Channel closed"))
        };
        
        // Generate!
        let _ = state.engine.generate_chat_stream(
            &prompt,
            max_tokens,
            temperature,
            false,
            callback,
        );
        
        // Send final [DONE] event
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });
    
    Sse::new(ReceiverStream::new(rx))
}

fn build_prompt_from_messages(messages: &[ChatMessage]) -> String {
    // Simple concatenation; for production, use your chat template!
    messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}

fn build_completion_response(report: GenerationReport) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: Uuid::new_v4().to_string(),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: "loci-local".to_string(),
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: report.text,
            },
            finish_reason: Some("stop".to_string()),
        }],
        usage: Some(Usage {
            prompt_tokens: 0,  // TODO: track from tokenizer
            completion_tokens: report.num_tokens,
            total_tokens: report.num_tokens,
        }),
    }
}