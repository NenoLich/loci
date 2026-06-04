use crate::inference::{SamplingResult, ToolFormatStyle};
use crate::api::types::{ChunkToolCall, LogprobsContent, ChatMessage, FinishReason, Usage, ToolCall, Role, CompletionTokensDetails};
use crate::error::LociError;

use candle_core::Device;

pub type StreamCallback = Box<dyn for<'a> FnMut(StreamFrame<'a>) -> anyhow::Result<()>>;

/// Events emitted by supervisors during token processing
#[derive(Clone, Debug)]
pub enum GenerationEvent {
    /// Tool call generation started
    ToolCallStarted,
    /// Tool call generation stopped
    ToolCallStopped,
    /// Reasoning generation started
    ReasoningStarted,
    /// Reasoning generation stopped
    ReasoningStopped,
    /// Tool call name being generated
    ToolCallNameChunk { chunk: ChunkToolCall },
    /// Tool call arguments being generated (streaming chunks)
    ToolCallArgumentsChunk { chunk: ChunkToolCall },
    /// Direct content text generated
    ContentSampled { sampling_result: SamplingResult },
    /// Reasoning content text generated
    ReasoningSampled { sampling_result: SamplingResult },
    /// Tokens to force in the stream
    ForceTokens { tokens: Vec<u32>},
    /// Eos reached
    GenerationStopped,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GenerationDataType {
    DirectContent,
    Reasoning,
    ToolCallName,
    ToolCallArguments,
}

pub struct StreamFrame<'a> {
    pub output: &'a str,
    pub tool_call_chunk: Option<ChunkToolCall>,
    pub output_type: GenerationDataType,
    pub logprobs: Option<LogprobsContent>,
}

pub struct GenerationReport {
    pub chat_message: ChatMessage,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    pub token_generation_sec: f64,
}

impl GenerationReport {
    pub fn new(
        content_text: &str, 
        reasoning_text: &str, 
        tool_calls: Option<Vec<ToolCall>>,
        finish_reason: FinishReason,
        prompt_tokens: u32,
        completion_tokens: u32,
        reasoning_tokens: u32,
        token_generation_sec: f64,
    ) -> Self {
        let reasoning_option = if reasoning_text.is_empty() { None } else { Some(reasoning_text) };
        let tool_calls_option = tool_calls.filter(|value| !value.is_empty());
        let chat_message = match (tool_calls_option, reasoning_option) {
            (None, None) => ChatMessage::new(Role::Assistant, content_text),
            (None, Some(reasoning)) => ChatMessage::with_reasoning_content(Role::Assistant, content_text, reasoning),
            (Some(tools), reasoning) => ChatMessage::with_tool_calls(Role::Assistant, tools, reasoning),
        };
        let usage = Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: None,
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens,
                ..Default::default()
            }),
        };
        Self {
            chat_message,
            finish_reason,
            usage,
            token_generation_sec,
        }
    }
}


pub struct PostSamplingConfig {
    pub tool_call_start_token_id: Option<u32>,
    pub tool_call_end_token_id: Option<u32>,
    pub reasoning_start_token_id: Option<u32>,
    pub reasoning_end_token_id: Option<u32>,
    pub tool_call_format_style: ToolFormatStyle,
    pub arg_key_open_token_id: Option<u32>,
    pub arg_key_close_token_id: Option<u32>,
    pub arg_value_open_token_id: Option<u32>,
    pub arg_value_close_token_id: Option<u32>,
}

pub struct DeviceManager;

impl DeviceManager {
    pub fn select() -> Result<Device, LociError> {
        if cfg!(feature = "cuda") && candle_core::utils::cuda_is_available() {
            tracing::debug!("Running on CUDA");
            Ok(Device::new_cuda(0).map_err(|e| LociError::ModelLoad(format!("CUDA device selection failed: {}", e)))?)
        } else {
            tracing::debug!("Running on CPU");
            Ok(Device::Cpu)
        }
    }
}
