use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Value, json};
use axum::extract::{FromRequest, Request};
use axum::response::{Response, IntoResponse};
use axum::http::StatusCode;
use axum::Json;

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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Display for Role {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: Role,
    #[serde(serialize_with = "serialize_option_string")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>, 
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn new(role: Role, content: &str) -> Self {
        Self {
            role,
            content: Some(content.to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn with_reasoning_content(role: Role, content: &str, reasoning_content: &str) -> Self {
        Self {
            role,
            content: Some(content.to_string()),
            reasoning_content: Some(reasoning_content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn with_tool_calls(role: Role, tool_calls: Vec<ToolCall>, reasoning_content: Option<&str>) -> Self {
        Self {
            role,
            content: None,
            reasoning_content: reasoning_content.map(|s| s.to_string()),
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Low,
    Medium,
    High,
}

impl Display for ReasoningEffort {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            ReasoningEffort::None => write!(f, "none"),
            ReasoningEffort::Low => write!(f, "low"),
            ReasoningEffort::Medium => write!(f, "medium"),
            ReasoningEffort::High => write!(f, "high"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}


impl Display for FinishReason {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            FinishReason::Stop => write!(f, "stop"),
            FinishReason::Length => write!(f, "length"),
            FinishReason::ToolCalls => write!(f, "tool_calls"),
            FinishReason::ContentFilter => write!(f, "content_filter"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamOptions {
    pub include_usage: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tool {
    pub r#type: String,
    pub function: Function,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Function {
    pub name: String,
    pub description: Option<String>,
    pub parameters: FunctionParameters,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionParameters {
    pub r#type: String,
    pub properties: Option<HashMap<String, Value>>,
    pub required: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ToolChoice {
    // Handles string flags: "none", "auto", "required"
    Mode(ToolChoiceMode),
    
    // Handles forcing a specific function call
    Specific(SpecificToolChoice),
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    None,
    Auto,
    Required,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct SpecificToolChoice {
    pub r#type: String, // Always "function"
    pub function: SpecificFunctionChoice,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct SpecificFunctionChoice {
    pub name: String, // The name of the specific function to force
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub arguments: HashMap<String, Value>, 
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

#[derive(Debug, Serialize, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Serialize, Clone, Default)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
    #[serde(default)]
    pub audio_tokens: u32,
}

#[derive(Debug, Serialize, Clone, Default)]
pub struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: u32,
    #[serde(default)]
    pub audio_tokens: u32,
    #[serde(default)]
    pub accepted_prediction_tokens: u32,
    #[serde(default)]
    pub rejected_prediction_tokens: u32,
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

#[derive(Clone, Debug, Serialize)]
pub struct ChunkToolCall {
    // The index of the tool call in the array
    pub index: u32,
    // Only present on the first chunk initiating the tool call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    // Only present on the first chunk initiating the tool call (always "function")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    pub function: ChunkFunctionCall,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChunkFunctionCall {
    // Only present on the first chunk initiating the function call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>, 
    // Essential: Pieces of the JSON argument string stream over time
    pub arguments: String, 
}

#[derive(Debug, Serialize)]
pub struct ChunkLogprob {
    pub content: Vec<LogprobsContent>,
}

#[derive(Debug, Serialize)]
pub struct LogprobsContent {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<Vec<TopLogprobs>>,
}

#[derive(Debug, Serialize)]
pub struct TopLogprobs {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
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
                ChatMessage::new(Role::System, "You are a helpfull assistant")
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

fn serialize_option_string<S>(value: &Option<String>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match value {
        Some(v) => serializer.serialize_str(v),
        None => serializer.serialize_str(""),
    }
}