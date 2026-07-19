use axum::Json;
use axum::extract::{FromRequest, Request};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::types::{
    ChatMessage, ChunkLogprob, ChunkToolCall, FinishReason, ReasoningEffort, Role, Tool,
    ToolChoice, ToolChoiceMode, Usage,
};

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

#[derive(Debug, Clone)]
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

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let body_string = String::from_request(req, state).await.map_err(|e| {
            let error = json!({
                "error": {
                    "message": format!("Failed to read request body: {}", e),
                    "type": "invalid_request_error"
                }
            });
            (StatusCode::BAD_REQUEST, Json(error)).into_response()
        })?;

        let payload: ChatCompletionRequest = serde_json::from_str(&body_string).map_err(|e| {
            // This will correctly catch missing fields like "model is required"
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
                if message.role == Role::User
                    && message
                        .content
                        .as_ref()
                        .is_some_and(|content| !content.is_empty())
                {
                    has_user_message = true;
                } else if message.role == Role::System
                    && message
                        .content
                        .as_ref()
                        .is_some_and(|content| !content.is_empty())
                {
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
                ChatMessage::new(Role::System, "You are a helpful assistant."),
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, routing::post};
    use axum_test::TestServer;
    use insta::assert_json_snapshot;
    use rstest::rstest;
    use serde_json::json;

    // A dummy handler that just echoes back success if validation passes
    async fn dummy_handler(
        ValidatedChatCompletionRequest { .. }: ValidatedChatCompletionRequest,
    ) -> &'static str {
        "ok"
    }

    fn test_app() -> TestServer {
        let router = Router::new().route("/v1/chat/completions", post(dummy_handler));
        TestServer::new(router)
    }

    #[tokio::test]
    async fn test_invalid_json_payload() {
        let server = test_app();

        // Send completely broken JSON (missing closing brace)
        let response = server
            .post("/v1/chat/completions")
            .content_type("application/json")
            .text("{ \"model\": \"gpt-4\" ")
            .await;

        // Force a 400 Bad Request
        response.assert_status(StatusCode::BAD_REQUEST);

        let json_body: serde_json::Value = serde_json::from_str(&response.text()).unwrap();
        assert_json_snapshot!(json_body, @r#"
        {
          "error": {
            "message": "Invalid JSON: EOF while parsing an object at line 1 column 19",
            "type": "invalid_request_error"
          }
        }
        "#);
    }

    #[tokio::test]
    async fn test_model_is_required() {
        let server = test_app();

        let response = server
            .post("/v1/chat/completions")
            .content_type("application/json")
            .text(json!({
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ],
                "not_model": "gpt-4"
            }))
            .await;

        response.assert_status(StatusCode::BAD_REQUEST);

        let json_body: serde_json::Value = serde_json::from_str(&response.text()).unwrap();
        assert_json_snapshot!(json_body, @r#"
        {
          "error": {
            "message": "model is required",
            "type": "invalid_request_error"
          }
        }
        "#);
    }

    #[tokio::test]
    async fn test_messages_are_present() {
        let server = test_app();

        let response = server
            .post("/v1/chat/completions")
            .content_type("application/json")
            .text(json!({
                "not_messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ],
                "model": "gpt-4"
            }))
            .await;

        response.assert_status(StatusCode::BAD_REQUEST);

        let json_body: serde_json::Value = serde_json::from_str(&response.text()).unwrap();
        assert_json_snapshot!(json_body, @r#"
        {
          "error": {
            "message": "Invalid JSON: missing field `messages` at line 1 column 68",
            "type": "invalid_request_error"
          }
        }
        "#);
    }

    #[tokio::test]
    async fn test_user_message_is_required() {
        let server = test_app();

        let response = server
            .post("/v1/chat/completions")
            .content_type("application/json")
            .text(json!({
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Hello"
                    }
                ],
                "model": "gpt-4"
            }))
            .await;

        response.assert_status(StatusCode::BAD_REQUEST);

        let json_body: serde_json::Value = serde_json::from_str(&response.text()).unwrap();
        assert_json_snapshot!(json_body, @r#"
        {
          "error": {
            "message": "messages must contain at least one user message",
            "type": "invalid_request_error"
          }
        }
        "#);
    }

    #[tokio::test]
    async fn test_system_message_injected_when_missing() {
        let req = Request::builder()
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(
                r#"{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}"#,
            ))
            .unwrap();

        let state = &();
        let result = ValidatedChatCompletionRequest::from_request(req, state)
            .await
            .expect("Extraction failed because of a structural or validation error");
        assert_eq!(result.messages[0].role, Role::System);
        assert_eq!(
            result.messages[0].content.as_ref().unwrap(),
            "You are a helpful assistant."
        );
    }

    #[rstest]
    #[case(
        r#"{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}"#,
        ToolChoice::Mode(ToolChoiceMode::None)
    )]
    #[case(
        r#"{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "tools": [{"type": "function", "function": {"name": "get_candidate_status", "description": "Retrieves the current status of a candidate in the recruitment process", "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string", "description": "Unique identifier for the candidate"}}, "required": ["candidate_id"]}}}]}"#,
        ToolChoice::Mode(ToolChoiceMode::Auto),
    )]
    #[case(
        r#"{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "tools": []}"#,
        ToolChoice::Mode(ToolChoiceMode::None)
    )]
    #[case(
        r#"{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "tool_choice": "auto"}"#,
        ToolChoice::Mode(ToolChoiceMode::Auto),
    )]
    #[case(
        r#"{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "tool_choice": "none", "tools": [{"type": "function", "function": {"name": "get_candidate_status", "description": "Retrieves the current status of a candidate in the recruitment process", "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string", "description": "Unique identifier for the candidate"}}, "required": ["candidate_id"]}}}]}"#,
        ToolChoice::Mode(ToolChoiceMode::None),
    )]
    #[tokio::test]
    async fn test_tool_choice_resolution(
        #[case] payload: &str,
        #[case] expected_tool_choice: ToolChoice,
    ) {
        let req = Request::builder()
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(payload.to_string()))
            .unwrap();

        let state = &();
        let result = ValidatedChatCompletionRequest::from_request(req, state)
            .await
            .expect("Extraction failed because of a structural or validation error");
        assert_eq!(result.tool_choice, expected_tool_choice);
    }

    #[rstest]
    #[case(
        r#"{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}"#,
        None
    )]
    #[case(
        r#"{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}"#,
        Some(100),
    )]
    #[case(
        r#"{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "max_completion_tokens": 100}"#,
        Some(100),
    )]
    #[tokio::test]
    async fn test_max_tokens_aliases(
        #[case] payload: &str,
        #[case] expected_max_tokens: Option<usize>,
    ) {
        let req = Request::builder()
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(payload.to_string()))
            .unwrap();

        let state = &();
        let result = ValidatedChatCompletionRequest::from_request(req, state)
            .await
            .expect("Extraction failed because of a structural or validation error");
        assert_eq!(result.max_tokens, expected_max_tokens);
    }

    #[rstest]
    #[case(
        r#"{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}"#,
        None
    )]
    #[case(
        r#"{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "repetition_penalty": 1.12}"#,
        Some(1.12),
    )]
    #[case(
        r#"{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "frequency_penalty": 1.14}"#,
        Some(1.14),
    )]
    #[tokio::test]
    async fn test_repetition_penalty_aliases(
        #[case] payload: &str,
        #[case] expected_repetition_penalty: Option<f32>,
    ) {
        let req = Request::builder()
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(payload.to_string()))
            .unwrap();

        let state = &();
        let result = ValidatedChatCompletionRequest::from_request(req, state)
            .await
            .expect("Extraction failed because of a structural or validation error");
        assert_eq!(result.repetition_penalty, expected_repetition_penalty);
    }

    #[rstest]
    #[case(
        r#"{"model": "modelgpt-4", "messages": [{"role": "user", "content": "Hello"}]}"#,
        "modelgpt-4"
    )]
    #[case(
        r#"{"model": "model\\gpt-4", "messages": [{"role": "user", "content": "Hello"}]}"#,
        "model/gpt-4"
    )]
    #[case(
        r#"{"model": "model\\\\gpt-4", "messages": [{"role": "user", "content": "Hello"}]}"#,
        "model//gpt-4"
    )]
    #[tokio::test]
    async fn test_normalized_model_name(#[case] payload: &str, #[case] expected_model: &str) {
        let result = ValidatedChatCompletionRequest::from_request(
            Request::builder()
                .header(axum::http::header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(payload.to_string()))
                .unwrap(),
            &(),
        )
        .await
        .expect("Extraction failed because of a structural or validation error");
        assert_eq!(result.model, expected_model);
    }
}
