use axum::response::sse::Event;
use serde_json::json;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::UNIX_EPOCH;
use tokio::sync::mpsc::error::TrySendError::{Closed, Full};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::oneshot;
use tokio::time::{Duration, Instant, sleep};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};
use uuid::Uuid;

use crate::api::types::{
    ChatCompletionChunk, ChatCompletionResponse, Choice, ChunkChoice, ChunkDelta,
    ValidatedChatCompletionRequest,
};
use crate::config::{GenerationOverrides, InferenceConfig, ModelCacheConfig};
use crate::error::LociError;
use crate::inference::{
    GenerationContext, GenerationDataType, GenerationReport, InferenceEngine, StreamCallback,
    StreamFrame,
};
use crate::types::{ChunkLogprob, FinishReason, Role};

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
    active_engine: Option<Arc<InferenceEngine>>,
    generation_context: Option<GenerationContext>,
    command_rx: Receiver<WorkerCommand>,
    idle_timeout: Duration,
    cancellation_token: CancellationToken,
}

impl EngineWorker {
    pub fn new(
        inference_config: InferenceConfig,
        model_cache_config: ModelCacheConfig,
        engine_option: Option<InferenceEngine>,
        command_rx: Receiver<WorkerCommand>,
        idle_timeout: u64,
        cancellation_token: CancellationToken,
    ) -> Result<Self, LociError> {
        let (active_engine, generation_context) = if let Some(engine) = engine_option {
            let model_name = engine.model_name();
            let cache_info = engine.model_cache_info();
            let ctx =
                GenerationContext::new(model_name, Some(model_cache_config.clone()), cache_info)?;
            (Some(Arc::new(engine)), Some(ctx))
        } else {
            (None, None)
        };

        Ok(EngineWorker {
            inference_config,
            model_cache_config,
            active_engine,
            generation_context,
            command_rx,
            idle_timeout: Duration::from_secs(idle_timeout),
            cancellation_token,
        })
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
                            let (_, req_model_name) = req.model.rsplit_once(['/', '\\']).unwrap_or(("", &req.model));
                            let current_model_match = self.active_engine.as_ref()
                                .map(|s| s.model_name() == req_model_name)
                                .unwrap_or(false);

                            if !current_model_match {
                                // Load new model (automatically drops old one)
                                self.drop_engine_and_ctx();
                                let engine_builder = InferenceEngine::builder()
                                    .with_gguf_metadata(req.model.clone())
                                    .inference_config(Some(self.inference_config.clone()));

                                let engine_result = tokio::task::spawn_blocking(move || {
                                    engine_builder.build()
                                }).await.unwrap();

                                match engine_result {
                                    Ok(eng) => {
                                        self.active_engine = Some(Arc::new(eng));
                                    }
                                    Err(e) => {
                                        let _ = response_tx.send(Err(e.to_string()));
                                        continue;
                                    }
                                }
                                if let Some(cache_info) = self.active_engine.as_ref().map(|s| s.model_cache_info()) {
                                    match GenerationContext::new(req_model_name, Some(self.model_cache_config.clone()), cache_info) {
                                        Ok(ctx) => {
                                            self.generation_context = Some(ctx);
                                        }
                                        Err(e) => {
                                            let _ = response_tx.send(Err(e.to_string()));
                                            continue;
                                        }
                                    }
                                }
                            }

                            let (stream_tx, stream_rx) = tokio::sync::mpsc::channel(128);

                            // Send the receiving end back to the Axum handler immediately
                            let _ = response_tx.send(Ok(stream_rx));

                            let Some(engine) = self.active_engine.clone() else { continue; };
                            let Some(mut ctx) = self.generation_context.take() else { continue; };

                            let generation_cancel_token = self.cancellation_token.clone();

                            let returned_ctx = tokio::task::spawn_blocking(move || {
                                if req.stream.unwrap_or(false) {
                                    run_stream_generation(&engine, &mut ctx, req, stream_tx, generation_cancel_token);
                                } else {
                                    run_single_generation(&engine, &mut ctx, req, stream_tx, generation_cancel_token);
                                }
                                ctx
                            }).await.unwrap();

                            self.generation_context = Some(returned_ctx);

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
                        self.drop_engine_and_ctx(); // Drops the engine and frees VRAM/RAM
                    }
                }
            }
        }

        self.drop_engine_and_ctx();
    }

    fn drop_engine_and_ctx(&mut self) {
        if let Some(ctx) = self.generation_context.as_mut() {
            ctx.save_active_cache();
        }
        self.generation_context = None;
        self.active_engine = None;
    }
}

fn run_stream_generation(
    engine: &InferenceEngine,
    ctx: &mut GenerationContext,
    req: ValidatedChatCompletionRequest,
    stream_tx: Sender<Result<Event, Infallible>>,
    cancellation_token: CancellationToken,
) {
    let overrides = GenerationOverrides::default()
        .with_temperature(req.temperature)
        .with_top_p(req.top_p)
        .with_max_tokens(req.max_tokens)
        .with_repetition_penalty(req.repetition_penalty)
        .with_tool_choice(Some(req.tool_choice))
        .with_reasoning_effort(req.reasoning_effort)
        .with_stop_tokens(req.stop)
        .with_logprobs(req.logprobs)
        .with_top_logprobs(req.top_logprobs)
        .with_seed(req.seed);

    let model_name = engine.model_name();
    // Setup initial chunk
    let static_data = StaticChatCompletionData {
        id: Uuid::new_v4().to_string(),
        created: std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: model_name.to_string(),
        system_fingerprint: format!("loci-{}", &model_name),
    };

    let initial_chunk = build_initial_chunk(static_data.clone());
    if let Ok(event) = Event::default().json_data(initial_chunk)
        && stream_tx.try_send(Ok(event)).is_err()
    {
        // The client disconnected (closed browser tab)
        info!("User disconnected. Stopping generation.");
        return;
    };

    let callback_tx = stream_tx.clone();
    let callback_static_data = static_data.clone();
    let callback: StreamCallback = Box::new(move |frame_data| {
        if cancellation_token.is_cancelled() {
            info!("Server shutdown signal received. Aborting generation immediately.");
            return Err(LociError::Stream(
                "Server shutdown signal received.".to_string(),
            ));
        }
        let regular_chunk = build_regular_chunk(callback_static_data.clone(), frame_data);
        let event = Event::default()
            .json_data(regular_chunk)
            .map_err(|e| LociError::Stream(e.to_string()))?;
        match callback_tx.try_send(Ok(event)) {
            Err(Full(message)) => {
                std::thread::sleep(std::time::Duration::from_micros(10));
                // Try one more time after the sleep
                if callback_tx.try_send(message).is_err() {
                    return Err(LociError::Stream(
                        "Stream channel backed up or closed".to_string(),
                    ));
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
    match engine.generate_chat_stream(
        &req.messages,
        &req.tools.unwrap_or_default(),
        ctx,
        overrides,
        callback,
    ) {
        Ok(report) => {
            debug!("Generation report: {:#?}", report);
            let final_content_chunk =
                build_final_content_chunk(static_data.clone(), report.finish_reason.clone());
            if let Ok(final_chunk_event) = Event::default().json_data(final_content_chunk)
                && stream_tx.try_send(Ok(final_chunk_event)).is_err()
            {
                // The client disconnected (closed browser tab)
                info!("User disconnected. Stopping generation.");
                return;
            }
            if matches!(req.stream_options, Some(options) if options.include_usage == Some(true)) {
                let usage_chunk = build_usage_chunk(static_data.clone(), report);
                if let Ok(usage_chunk_event) = Event::default().json_data(usage_chunk)
                    && stream_tx.try_send(Ok(usage_chunk_event)).is_err()
                {
                    // The client disconnected (closed browser tab)
                    info!("User disconnected. Stopping generation.");
                }
            }
        }
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
                if let Ok(event) = Event::default().json_data(json_error)
                    && stream_tx.try_send(Ok(event)).is_err()
                {
                    // The client disconnected (closed browser tab)
                    info!("User disconnected. Stopping generation.");
                };
            }
        }
    }
}

fn build_initial_chunk(static_data: StaticChatCompletionData) -> ChatCompletionChunk {
    let choices = vec![ChunkChoice {
        index: 0,
        delta: ChunkDelta {
            role: Some(Role::Assistant),
            content: Some(String::from("")),
            reasoning_content: None,
            tool_calls: None,
        },
        logprobs: None,
        finish_reason: None,
    }];
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
            tool_calls: frame_data.tool_call_chunk.map(|tc| vec![tc]),
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
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            logprobs: frame_data
                .logprobs
                .map(|cl| ChunkLogprob { content: vec![cl] }),
            finish_reason: None,
        }],
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
    engine: &InferenceEngine,
    ctx: &mut GenerationContext,
    req: ValidatedChatCompletionRequest,
    stream_tx: Sender<Result<Event, Infallible>>,
    cancellation_token: CancellationToken,
) {
    let overrides = GenerationOverrides::default()
        .with_temperature(req.temperature)
        .with_top_p(req.top_p)
        .with_max_tokens(req.max_tokens)
        .with_repetition_penalty(req.repetition_penalty)
        .with_tool_choice(Some(req.tool_choice))
        .with_reasoning_effort(req.reasoning_effort)
        .with_stop_tokens(req.stop)
        .with_logprobs(req.logprobs)
        .with_top_logprobs(req.top_logprobs)
        .with_seed(req.seed);

    let model_name = engine.model_name();
    let callback_tx = stream_tx.clone();
    let callback: StreamCallback = Box::new(move |_| {
        if callback_tx.is_closed() {
            info!("User disconnected. Stopping generation.");
            return Err(LociError::Stream("User disconnected".to_string()));
        }
        if cancellation_token.is_cancelled() {
            info!("Server shutdown signal received. Aborting generation immediately.");
            return Err(LociError::Stream(
                "Server shutdown signal received.".to_string(),
            ));
        }
        Ok(())
    });
    match engine.generate_chat_stream(
        &req.messages,
        &req.tools.unwrap_or_default(),
        ctx,
        overrides,
        callback,
    ) {
        Ok(report) => {
            debug!("Generation report: {:#?}", report);
            let chat_completion_response = build_chat_completion_response(model_name, report);
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
                    }
                    _ => (),
                };
            }
        }
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
                if let Ok(event) = Event::default().json_data(json_error)
                    && stream_tx.try_send(Ok(event)).is_err()
                {
                    // The client disconnected (closed browser tab)
                    info!("User disconnected. Stopping generation.");
                };
            }
        }
    }
}

fn build_chat_completion_response(model: &str, report: GenerationReport) -> ChatCompletionResponse {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ChatMessage, ChunkFunctionCall, ChunkToolCall, CompletionTokensDetails,
        PromptTokensDetails, Usage, ModelCacheFragmentation
    };
    use insta::assert_json_snapshot;
    use candle_core::Device;

    use crate::tokenizer::MockTokenizer;
    use crate::model::model_base::{MockModel, ModelCacheType};
    use crate::inference::PostSamplingConfig;
    use crate::config::GenerationConfig;
    use crate::inference::model_cache::{MockCacheLoader, MockModelCacheManagerInterface};

    fn mock_static_chat_completions_data() -> StaticChatCompletionData {
        StaticChatCompletionData {
            id: "1234".to_string(),
            created: 1234567u64,
            model: "test".to_string(),
            system_fingerprint: "loci-test".to_string(),
        }
    }

    fn mock_generation_report() -> GenerationReport {
        GenerationReport {
            chat_message: ChatMessage {
                role: Role::Assistant,
                content: Some("content".to_string()),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: FinishReason::Length,
            usage: Usage {
                prompt_tokens: 51,
                completion_tokens: 79,
                total_tokens: 130,
                prompt_tokens_details: Some(PromptTokensDetails {
                    cached_tokens: 0,
                    audio_tokens: 0,
                }),
                completion_tokens_details: Some(CompletionTokensDetails::default()),
            },
            token_generation_sec: 2.0,
        }
    }

    #[test]
    fn test_build_initial_chunk() {
        let data = mock_static_chat_completions_data();
        let response = build_initial_chunk(data);

        assert_json_snapshot!(response, @r#"
        {
          "id": "1234",
          "object": "chat.completion.chunk",
          "created": 1234567,
          "model": "test",
          "system_fingerprint": "loci-test",
          "choices": [
            {
              "index": 0,
              "delta": {
                "role": "assistant",
                "content": ""
              }
            }
          ],
          "usage": null
        }
        "#);
    }

    #[test]
    fn test_build_regular_chunk_with_direct_content() {
        let data = mock_static_chat_completions_data();
        let frame_data = StreamFrame {
            output: "content",
            output_type: GenerationDataType::DirectContent,
            tool_call_chunk: None,
            logprobs: None,
        };
        let response = build_regular_chunk(data, frame_data);

        assert_json_snapshot!(response, @r#"
        {
          "id": "1234",
          "object": "chat.completion.chunk",
          "created": 1234567,
          "model": "test",
          "system_fingerprint": "loci-test",
          "choices": [
            {
              "index": 0,
              "delta": {
                "content": "content"
              }
            }
          ],
          "usage": null
        }
        "#);
    }

    #[test]
    fn test_build_regular_chunk_with_reasoning() {
        let data = mock_static_chat_completions_data();
        let frame_data = StreamFrame {
            output: "content",
            output_type: GenerationDataType::Reasoning,
            tool_call_chunk: None,
            logprobs: None,
        };
        let response = build_regular_chunk(data, frame_data);

        assert_json_snapshot!(response, @r#"
        {
          "id": "1234",
          "object": "chat.completion.chunk",
          "created": 1234567,
          "model": "test",
          "system_fingerprint": "loci-test",
          "choices": [
            {
              "index": 0,
              "delta": {
                "reasoning_content": "content"
              }
            }
          ],
          "usage": null
        }
        "#);
    }

    #[test]
    fn test_build_regular_chunk_with_tool_call_name() {
        let data = mock_static_chat_completions_data();
        let frame_data = StreamFrame {
            output: "",
            output_type: GenerationDataType::ToolCallName,
            tool_call_chunk: Some(ChunkToolCall {
                index: 0,
                id: Some(String::from("4321")),
                r#type: Some(String::from("function")),
                function: ChunkFunctionCall {
                    name: Some(String::from("test_tool")),
                    arguments: String::new(),
                },
            }),
            logprobs: None,
        };
        let response = build_regular_chunk(data, frame_data);

        assert_json_snapshot!(response, @r#"
        {
          "id": "1234",
          "object": "chat.completion.chunk",
          "created": 1234567,
          "model": "test",
          "system_fingerprint": "loci-test",
          "choices": [
            {
              "index": 0,
              "delta": {
                "tool_calls": [
                  {
                    "index": 0,
                    "id": "4321",
                    "type": "function",
                    "function": {
                      "name": "test_tool",
                      "arguments": ""
                    }
                  }
                ]
              }
            }
          ],
          "usage": null
        }
        "#);
    }

    #[test]
    fn test_build_regular_chunk_with_tool_call_arguments() {
        let data = mock_static_chat_completions_data();
        let frame_data = StreamFrame {
            output: "",
            output_type: GenerationDataType::ToolCallArguments,
            tool_call_chunk: Some(ChunkToolCall {
                index: 0,
                id: None,
                r#type: None,
                function: ChunkFunctionCall {
                    name: None,
                    arguments: String::from("[\"arg1\", \"arg2\"]"),
                },
            }),
            logprobs: None,
        };
        let response = build_regular_chunk(data, frame_data);

        assert_json_snapshot!(response, @r#"
        {
          "id": "1234",
          "object": "chat.completion.chunk",
          "created": 1234567,
          "model": "test",
          "system_fingerprint": "loci-test",
          "choices": [
            {
              "index": 0,
              "delta": {
                "tool_calls": [
                  {
                    "index": 0,
                    "function": {
                      "arguments": "[\"arg1\", \"arg2\"]"
                    }
                  }
                ]
              }
            }
          ],
          "usage": null
        }
        "#);
    }

    #[test]
    fn test_build_final_content_chunk_with_finish_reason_stop() {
        let data = mock_static_chat_completions_data();
        let response = build_final_content_chunk(data, FinishReason::Stop);

        assert_json_snapshot!(response, @r#"
        {
          "id": "1234",
          "object": "chat.completion.chunk",
          "created": 1234567,
          "model": "test",
          "system_fingerprint": "loci-test",
          "choices": [
            {
              "index": 0,
              "delta": {},
              "finish_reason": "stop"
            }
          ],
          "usage": null
        }
        "#);
    }

    #[test]
    fn test_build_final_content_chunk_with_finish_reason_length() {
        let data = mock_static_chat_completions_data();
        let response = build_final_content_chunk(data, FinishReason::Length);

        assert_json_snapshot!(response, @r#"
        {
          "id": "1234",
          "object": "chat.completion.chunk",
          "created": 1234567,
          "model": "test",
          "system_fingerprint": "loci-test",
          "choices": [
            {
              "index": 0,
              "delta": {},
              "finish_reason": "length"
            }
          ],
          "usage": null
        }
        "#);
    }

    #[test]
    fn test_build_final_content_chunk_with_finish_reason_tool_calls() {
        let data = mock_static_chat_completions_data();
        let response = build_final_content_chunk(data, FinishReason::ToolCalls);

        assert_json_snapshot!(response, @r#"
        {
          "id": "1234",
          "object": "chat.completion.chunk",
          "created": 1234567,
          "model": "test",
          "system_fingerprint": "loci-test",
          "choices": [
            {
              "index": 0,
              "delta": {},
              "finish_reason": "tool_calls"
            }
          ],
          "usage": null
        }
        "#);
    }

    #[test]
    fn test_build_usage_chunk() {
        let data = mock_static_chat_completions_data();
        let report = mock_generation_report();

        let response = build_usage_chunk(data, report);

        assert_json_snapshot!(response, @r#"
        {
          "id": "1234",
          "object": "chat.completion.chunk",
          "created": 1234567,
          "model": "test",
          "system_fingerprint": "loci-test",
          "choices": [],
          "usage": {
            "prompt_tokens": 51,
            "completion_tokens": 79,
            "total_tokens": 130,
            "prompt_tokens_details": {
              "cached_tokens": 0,
              "audio_tokens": 0
            },
            "completion_tokens_details": {
              "reasoning_tokens": 0,
              "audio_tokens": 0,
              "accepted_prediction_tokens": 0,
              "rejected_prediction_tokens": 0
            }
          }
        }
        "#);
    }

    #[test]
    fn test_build_chat_completion_response() {
        let report = mock_generation_report();
        let response = build_chat_completion_response("test-model", report);
        let mut value = serde_json::to_value(&response).unwrap();
        value.as_object_mut().unwrap().remove("id");
        value.as_object_mut().unwrap().remove("created");
        assert_json_snapshot!(value, @r#"
        {
          "usage": {
            "prompt_tokens": 51,
            "completion_tokens": 79,
            "total_tokens": 130,
            "prompt_tokens_details": {
              "cached_tokens": 0,
              "audio_tokens": 0
            },
            "completion_tokens_details": {
              "reasoning_tokens": 0,
              "audio_tokens": 0,
              "accepted_prediction_tokens": 0,
              "rejected_prediction_tokens": 0
            }
          },
          "object": "chat.completion",
          "choices": [
            {
              "index": 0,
              "message": {
                "role": "assistant",
                "content": "content"
              },
              "finish_reason": "length"
            }
          ],
          "model": "test-model",
          "system_fingerprint": "loci-test-model"
        }
        "#);
    }

    #[tokio::test]
    async fn test_drop_engine_due_to_timeout() {
        tokio::time::pause();

        let inference_config = InferenceConfig::default();
        let model_cache_config = ModelCacheConfig::default();
        let tokenizer = Box::new(MockTokenizer::new());
        let model = Box::new(MockModel::new());
        let post_sampling_config = PostSamplingConfig::default();
        let gen_builder = GenerationConfig::builder();
        let active_engine = Arc::new(InferenceEngine {
            tokenizer,
            model,
            device: Device::Cpu,
            model_name: "mock".to_string(),
            vocab_size: 100,
            flash_attn: false,
            supports_reasoning: false,
            supports_tool_calling: false,
            flatten_tools_to_functions: true,
            post_sampling_config,
            gen_builder,
        });
        let generation_context = GenerationContext {
            model: "mock".to_string(),
            token_ids: vec![],
            active_cache: vec![],
            model_layers_count: 2,
            model_cache_seq_len_dim: 2,
            model_cache_fragmentation: ModelCacheFragmentation::TokenWise,
            block_boundary_conv_cache: vec![],
            prefix_caching: false,
            cache_type: ModelCacheType::FullAttn,
            cache_manager: Box::new(MockModelCacheManagerInterface::new()),
            cache_loader: Box::new(MockCacheLoader::new()),
            cache_metadata: vec![],
        };
        let (_command_tx, command_rx) = tokio::sync::mpsc::channel(1);

        let mut worker = EngineWorker {
            inference_config,
            model_cache_config,
            active_engine: Some(active_engine),
            generation_context: Some(generation_context),
            command_rx,
            idle_timeout: Duration::from_secs(600),
            cancellation_token: CancellationToken::new(),
        };

        assert!(worker.active_engine.is_some());
        assert!(worker.generation_context.is_some());

        // Simulate what run() does: record timestamp, then sleep inside select.
        let last_used = Instant::now();

        // Fast-forward past the idle timeout + one sleep cycle.
        tokio::time::advance(Duration::from_secs(630)).await;

        // Simulate one iteration of the select loop in run().
        let timer = tokio::time::sleep(Duration::from_secs(30));
        tokio::pin!(timer);
        tokio::select! {
            biased;
            _ = timer.as_mut() => {
                if worker.active_engine.is_some() && last_used.elapsed() > worker.idle_timeout {
                    worker.drop_engine_and_ctx();
                }
            }
            _ = worker.command_rx.recv() => {}
        }

        assert!(worker.active_engine.is_none());
        assert!(worker.generation_context.is_none());
    }
}
