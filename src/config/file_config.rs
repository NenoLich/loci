use serde::{Deserialize};
use std::fmt::{self, Display, Formatter};
use std::path::Path;

use crate::api::types::{ToolChoice, ReasoningEffort};
use crate::error::LociError;

#[derive(Debug, Copy, Clone, clap::ValueEnum, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputeDtype {
    F32,
    F16
}

impl Display for ComputeDtype {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            ComputeDtype::F16 => write!(f, "f16"),
            ComputeDtype::F32 => write!(f, "f32"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct FileConfig {
    pub generation_config: Option<GenerationFileConfig>,
    pub inference_config: Option<InferenceFileConfig>,
    pub cache_config: Option<CacheFileConfig>,
}

impl FileConfig {
    pub fn load(filename: impl AsRef<Path>) -> Result<Self, LociError> {
        let config = std::fs::read_to_string(filename)
            .map_err(|e| LociError::Io(e))?;
        Ok(toml::from_str(&config).map_err(|e| LociError::Config(e.to_string()))?)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerationFileConfig {
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub tool_choice: Option<ToolChoice>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub stop_tokens: Option<Vec<String>>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<usize>,
    pub seed: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InferenceFileConfig {
    pub dtype: Option<ComputeDtype>,
    pub max_seq_len: Option<usize>,
    pub conv_on_cpu: Option<bool>,
    pub flash_attn: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CacheFileConfig;
