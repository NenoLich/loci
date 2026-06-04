use anyhow::{Error, Ok, Result, bail};
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

use crate::gguf::GgufInfo;
use crate::config::{Lfm2Parser, Deepseek2Parser};
use crate::inference::ToolFormatStyle;

#[derive(Debug, Clone)]
pub enum ModelArchitecture {
    Lfm2,
    Deepseek2,
}

impl FromStr for ModelArchitecture {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "lfm2" => Ok(ModelArchitecture::Lfm2),
            "deepseek2" => Ok(ModelArchitecture::Deepseek2),
            _ => bail!("Unsupported model architecture: {}", s),
        }
    }
}

impl fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ModelArchitecture::Lfm2 => write!(f, "Lfm2"),
            ModelArchitecture::Deepseek2 => write!(f, "Deepseek2"),
        }
    }
}

#[derive(Clone)]
pub struct ModelConfig {
    pub file_path: PathBuf,
    pub architecture: ModelArchitecture,
    pub hidden_size: usize,
    pub n_heads: usize,
    pub n_kv_heads: Vec<usize>,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub intermediate_ffn_size: usize,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub rms_epsilon: f32,
    pub conv_l_cache: Option<usize>,
    pub n_expert_used: Option<usize>,
    pub n_expert_group: Option<usize>,
    pub n_expert_group_used: Option<usize>,
    pub expert_gating_func: Option<usize>,
    pub n_leading_dense_block: Option<usize>,
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: Option<usize>,
    pub attn_key_length: Option<usize>,
    pub attn_value_length: Option<usize>,
    pub attn_key_length_mla: Option<usize>,
    pub attn_value_length_mla: Option<usize>,
    pub expert_ffn_size: Option<usize>,
    pub n_experts: Option<usize>,
    pub n_expert_shared: Option<usize>,
    pub expert_weights_scale: Option<f32>,
    pub expert_weights_norm: Option<bool>,
    pub n_rope_dims: Option<usize>,
    pub supports_tool_calling: bool,
    pub supports_reasoning: bool,
    pub tool_call_start_token_id: Option<u32>,
    pub tool_call_end_token_id: Option<u32>,
    pub flatten_tools_to_functions: bool,
    pub reasoning_start_token_id: Option<u32>,
    pub reasoning_end_token_id: Option<u32>,
    pub tool_call_format_style: ToolFormatStyle,
    pub arg_key_open_token_id: Option<u32>,
    pub arg_key_close_token_id: Option<u32>,
    pub arg_value_open_token_id: Option<u32>,
    pub arg_value_close_token_id: Option<u32>,
}

impl ModelConfig {
    pub fn from_gguf_info(gguf_info: &GgufInfo) -> Result<Self> {
        let gguf_meta = &gguf_info.kv_meta;
        let arch_str = gguf_meta
            .iter()
            .find(|&entry| entry.key == "general.architecture")
            .and_then(|m| m.value.as_string())
            .ok_or_else(|| anyhow::anyhow!("Missing general.architecture"))?;

        let arch = ModelArchitecture::from_str(&arch_str)?;

        match arch {
            ModelArchitecture::Lfm2 => Lfm2Parser::parse(gguf_info),
            ModelArchitecture::Deepseek2 => Deepseek2Parser::parse(gguf_info),
            _ => bail!("Unsupported model architecture: {}", arch),
        }
    }
}
