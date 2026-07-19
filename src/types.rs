use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

use std::collections::HashMap;
use std::fmt::{self, Debug, Display, Formatter};

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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
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

    pub fn with_tool_calls(
        role: Role,
        content: &str,
        tool_calls: Vec<ToolCall>,
        reasoning_content: Option<&str>,
    ) -> Self {
        Self {
            role,
            content: Some(content.to_string()),
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

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
}

impl Display for FinishReason {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            FinishReason::Stop => write!(f, "stop"),
            FinishReason::Length => write!(f, "length"),
            FinishReason::ToolCalls => write!(f, "tool_calls"),
        }
    }
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
    pub properties: Option<serde_json::Map<String, Value>>,
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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
pub struct FunctionDefinition {
    pub name: String,
    pub arguments: HashMap<String, Value>,
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

#[derive(Clone, Debug, Serialize, PartialEq, Default)]
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

#[derive(Clone, Debug, Serialize, PartialEq, Default)]
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

#[derive(Debug, Serialize, Clone, PartialEq)]
pub struct LogprobsContent {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<Vec<TopLogprobs>>,
}

#[derive(Debug, Serialize, Clone, PartialEq)]
pub struct TopLogprobs {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Serialize, PartialEq)]
pub enum ModelCacheFragmentation {
    BlockWise { block_size: usize },
    TokenWise,
}

impl Debug for ModelCacheFragmentation {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ModelCacheFragmentation::BlockWise { block_size } => {
                write!(f, "BlockWise({block_size})")
            }
            ModelCacheFragmentation::TokenWise => write!(f, "TokenWise"),
        }
    }
}

impl Display for ModelCacheFragmentation {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ModelCacheFragmentation::BlockWise { block_size } => {
                write!(f, "BlockWise({block_size})")
            }
            ModelCacheFragmentation::TokenWise => write!(f, "TokenWise"),
        }
    }
}

impl<'de> Deserialize<'de> for ModelCacheFragmentation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct FragmentationVisitor;
        impl<'de> serde::de::Visitor<'de> for FragmentationVisitor {
            type Value = ModelCacheFragmentation;

            fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
                formatter
                    .write_str("a fragmentation format like \"BlockWise(32)\" or \"TokenWise\"")
            }

            fn visit_str<E: serde::de::Error>(
                self,
                value: &str,
            ) -> Result<ModelCacheFragmentation, E> {
                if value == "TokenWise" {
                    return Ok(ModelCacheFragmentation::TokenWise);
                }
                if let Some(rest) = value
                    .strip_prefix("BlockWise(")
                    .and_then(|s| s.strip_suffix(')'))
                    && let Ok(block_size) = rest.parse::<usize>()
                {
                    return Ok(ModelCacheFragmentation::BlockWise { block_size });
                }
                Err(serde::de::Error::custom(format!(
                    "invalid fragmentation format: {}",
                    value
                )))
            }
        }
        deserializer.deserialize_str(FragmentationVisitor)
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

#[cfg(test)]
mod tests_top_level {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(Role::System, "system")]
    #[case(Role::User, "user")]
    #[case(Role::Assistant, "assistant")]
    #[case(Role::Tool, "tool")]
    fn test_role_display(#[case] role: Role, #[case] expected: &str) {
        assert_eq!(role.to_string(), expected);
    }

    #[rstest]
    #[case(FinishReason::Stop, "stop")]
    #[case(FinishReason::Length, "length")]
    #[case(FinishReason::ToolCalls, "tool_calls")]
    fn test_finish_reason_display(#[case] finish_reason: FinishReason, #[case] expected: &str) {
        assert_eq!(finish_reason.to_string(), expected);
    }

    #[rstest]
    #[case(ReasoningEffort::Low, "low")]
    #[case(ReasoningEffort::Medium, "medium")]
    #[case(ReasoningEffort::High, "high")]
    fn test_reasoning_effort_display(
        #[case] reasoning_effort: ReasoningEffort,
        #[case] expected: &str,
    ) {
        assert_eq!(reasoning_effort.to_string(), expected);
    }

    #[rstest]
    #[case(r#""BlockWise(32)""#, ModelCacheFragmentation::BlockWise { block_size: 32 })]
    #[case(r#""TokenWise""#, ModelCacheFragmentation::TokenWise)]
    fn test_fragmentaion_deserialize_success(
        #[case] fragmentation_str: &str,
        #[case] expected: ModelCacheFragmentation,
    ) {
        let deserialized: ModelCacheFragmentation =
            serde_json::from_str(fragmentation_str).unwrap();
        assert_eq!(deserialized, expected);
    }

    #[rstest]
    #[case(r#""""#, "invalid fragmentation format: ")]
    #[case(r#""BlockWise""#, "invalid fragmentation format: BlockWise")]
    fn test_fragmentaion_deserialize_failure(
        #[case] fragmentation_str: &str,
        #[case] expected_msg: &str,
    ) {
        let deserialized: Result<ModelCacheFragmentation, _> =
            serde_json::from_str(fragmentation_str);
        assert!(deserialized.unwrap_err().to_string().contains(expected_msg));
    }

    #[rstest]
    #[case(Role::System, "system")]
    #[case(Role::User, "user")]
    #[case(Role::Assistant, "assistant")]
    #[case(Role::Tool, "tool")]
    fn test_chat_message_new(#[case] role: Role, #[case] content: &str) {
        let chat_message = ChatMessage::new(role.clone(), content);
        assert_eq!(chat_message.role, role);
        assert_eq!(chat_message.content, Some(content.to_string()));
        assert_eq!(chat_message.reasoning_content, None);
        assert_eq!(chat_message.tool_calls, None);
        assert_eq!(chat_message.tool_call_id, None);
    }

    #[rstest]
    #[case(Role::System, "content", "")]
    #[case(Role::User, "content", "")]
    #[case(Role::Assistant, "content", "reasoning content")]
    #[case(Role::Tool, "content", "")]
    fn test_chat_message_with_reasoning_content(
        #[case] role: Role,
        #[case] content: &str,
        #[case] reasoning_content: &str,
    ) {
        let chat_message =
            ChatMessage::with_reasoning_content(role.clone(), content, reasoning_content);
        assert_eq!(chat_message.role, role);
        assert_eq!(chat_message.content, Some(content.to_string()));
        assert_eq!(
            chat_message.reasoning_content,
            Some(reasoning_content.to_string())
        );
        assert_eq!(chat_message.tool_calls, None);
        assert_eq!(chat_message.tool_call_id, None);
    }

    #[rstest]
    #[case(Role::System, "content", vec![], None)]
    #[case(Role::User, "content", vec![], None)]
    #[case(Role::Assistant, "content", vec![], Some("reasoning content"))]
    #[case(Role::Assistant, "content", vec![ToolCall {
        id: "id".to_string(),
        r#type: "function".to_string(),
        function: FunctionDefinition {
            name: "name".to_string(),
            arguments: HashMap::from([
                ("arg1".to_string(), serde_json::Value::String("arg1_value".to_string())),
                ("arg2".to_string(), serde_json::Value::String("arg2_value".to_string())),
            ])
        },
    }], Some("reasoning content"))]
    fn test_chat_message_with_tool_calls(
        #[case] role: Role,
        #[case] content: &str,
        #[case] tool_calls: Vec<ToolCall>,
        #[case] reasoning_content: Option<&str>,
    ) {
        let chat_message = ChatMessage::with_tool_calls(
            role.clone(),
            content,
            tool_calls.clone(),
            reasoning_content.clone(),
        );
        assert_eq!(chat_message.role, role);
        assert_eq!(chat_message.content, Some(content.to_string()));
        assert_eq!(
            chat_message.reasoning_content,
            reasoning_content.map(|s| s.to_string())
        );
        assert_eq!(chat_message.tool_calls, Some(tool_calls));
        assert_eq!(chat_message.tool_call_id, None);
    }

    #[test]
    fn test_usage_default() {
        let usage = Usage::default();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn test_prompt_token_details_default() {
        let prompt_token_details = PromptTokensDetails::default();
        assert_eq!(prompt_token_details.cached_tokens, 0);
        assert_eq!(prompt_token_details.audio_tokens, 0);
    }

    #[test]
    fn test_completion_token_details_default() {
        let completion_token_details = CompletionTokensDetails::default();
        assert_eq!(completion_token_details.reasoning_tokens, 0);
        assert_eq!(completion_token_details.audio_tokens, 0);
        assert_eq!(completion_token_details.accepted_prediction_tokens, 0);
        assert_eq!(completion_token_details.rejected_prediction_tokens, 0);
    }
}
