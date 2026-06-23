use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Value, json};
use axum::extract::{FromRequest, Request};
use axum::response::{Response, IntoResponse};
use axum::http::StatusCode;
use axum::Json;

use crate::types::{Role, ChatMessage, ReasoningEffort, Tool, ToolChoice, ToolChoiceMode, ChunkToolCall, ChunkLogprob, FinishReason, Usage};

use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(rename = "max_tokens", alias = "max_completion_tokens")]
    pub max_tokens: Option<usize>,
    #[serde(rename = "repetition_penalty", alias = "frequency_penalty")]
    pub repetition_penalty: Option<f32>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub stream: Option<bool>,
    pub stream_options: Option<StreamOptions>,
    pub stop: Option<Vec<String>>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<usize>,
    pub seed: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamOptions {
    pub include_usage: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String, 
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<FinishReason>,
}

// Streaming response (Server-Sent Events)
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String, 
    // In the last chunk (if include_usage is true), this array will be EMPTY []
    pub choices: Vec<ChunkChoice>,
    // Only populated on the very last chunk when stream_options.include_usage = true
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChunkLogprob>,
    // Omitted on running tokens; becomes Some("stop"), Some("tool_calls"), etc. at the end
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Serialize)]
pub struct ChunkDelta {
    // Role is only sent in the first chunk
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    // Text contents streamed token by token
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    // If the model is executing a tool, parts stream here over time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChunkToolCall>>,
}

pub struct ValidatedChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: ToolChoice,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub stream: Option<bool>,
    pub stream_options: Option<StreamOptions>,
    pub stop: Option<Vec<String>>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<usize>,
    pub seed: Option<usize>,
}

impl<S> FromRequest<S> for ValidatedChatCompletionRequest 
where 
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(
        req: Request,
        state: &S,
    ) -> Result<Self, Self::Rejection>
    {
        let Json(payload) = Json::<ChatCompletionRequest>::from_request(req, state)
            .await
            .map_err(|e| {
                let error = json!({
                    "error": {
                        "message": format!("Invalid JSON: {}", e),
                        "type": "invalid_request_error"
                    }
                });
                (StatusCode::BAD_REQUEST, Json(error)).into_response()
            })?;

        let Some(model_name) = payload.model else {
            let error = json!({
                "error": {
                    "message": "model is required",
                    "type": "invalid_request_error"
                }
            });
            return Err((StatusCode::BAD_REQUEST, Json(error)).into_response());
        };

        let model = model_name.replace('\\', "/");

        let mut messages = payload.messages;
        let mut has_user_message = false;
        let mut has_system_message = false;

        if !messages.is_empty() {
            for message in &messages {
                if message.role == Role::User && message.content.as_ref().is_some_and(|content| !content.is_empty()) {
                    has_user_message = true;
                } else if message.role == Role::System && message.content.as_ref().is_some_and(|content| !content.is_empty()) {
                    has_system_message = true;
                }
            }
        }

        if !has_user_message {
            let error = json!({
                "error": {
                    "message": "messages must contain at least one user message",
                    "type": "invalid_request_error"
                }
            });
            return Err((StatusCode::BAD_REQUEST, Json(error)).into_response());
        }

        if !has_system_message {
            messages.insert(
                0, 
                ChatMessage::new(Role::System, "You are a helpful assistant")
            );
        }

        let tool_choice = if let Some(tool_choice) = payload.tool_choice {
            tool_choice
        } else if payload.tools.is_some() && !payload.tools.as_ref().unwrap().is_empty() {
            ToolChoice::Mode(ToolChoiceMode::Auto)
        } else {
            ToolChoice::Mode(ToolChoiceMode::None)
        };

        let ChatCompletionRequest {
            tools,
            temperature,
            top_p,
            max_tokens,
            repetition_penalty,
            reasoning_effort,
            stream,
            stream_options,
            stop,
            logprobs,
            top_logprobs,
            seed,
            ..
        } = payload;

        Ok(ValidatedChatCompletionRequest {
            model,
            messages,
            tools,
            tool_choice,
            temperature,
            top_p,
            max_tokens,
            repetition_penalty,
            reasoning_effort,
            stream,
            stream_options,
            stop,
            logprobs,
            top_logprobs,
            seed,
        })
    }
}
