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

#[derive(Debug, Clone, Serialize)]
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

#[derive(Clone, Serialize)]
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
