use std::path::PathBuf;
use std::str::FromStr;

use crate::gguf::{GgufInfo, GgufKVMeta, GgufValue};
use crate::config::{ModelConfig, ModelArchitecture}; 
use crate::inference::ToolFormatStyle;

pub struct Lfm2ExtraParameters;

impl Lfm2ExtraParameters {
    pub const CACHE_SEQ_LEN_DIM: usize = 2;
    pub const SUPPORTS_TOOL_CALLING: bool = true;
    pub const SUPPORTS_REASONING: bool = false;
    pub const TOOL_CALL_START_TOKEN_ID: Option<u32> = Some(10);
    pub const TOOL_CALL_END_TOKEN_ID: Option<u32> = Some(11);
    pub const FLATTEN_TOOLS_TO_FUNCTIONS: bool = true;
    pub const REASONING_START_TOKEN_ID: Option<u32> = None;
    pub const REASONING_END_TOKEN_ID: Option<u32> = None;
    pub const TOOL_CALL_FORMAT_STYLE: ToolFormatStyle = ToolFormatStyle::PythonCall;
}

pub struct Lfm2Parser;

impl Lfm2Parser {
    pub fn parse(gguf_info: &GgufInfo) -> anyhow::Result<ModelConfig> {
        let architecture = ModelArchitecture::Lfm2;
        let mut model_name = None;
        let mut hidden_size = None;
        let mut n_heads = None;
        let mut n_kv_heads = None;
        let mut n_layers = None;
        let mut vocab_size = None;
        let mut intermediate_ffn_size = None;
        let mut rope_theta = None;
        let mut max_seq_len = None;
        let mut rms_epsilon = 1e-5;
        let mut conv_l_cache = None;

        let metadata = &gguf_info.kv_meta;
        for entry in metadata {
            match entry.key.as_str() {
                "general.name" => model_name = entry.value.as_string(),
                "lfm2.embedding_length" => hidden_size = entry.value.as_usize(),
                "lfm2.attention.head_count" => n_heads = entry.value.as_usize(),
                "lfm2.attention.head_count_kv" => n_kv_heads = Some(Self::build_n_kv_count(entry)?),
                "lfm2.block_count" => n_layers = entry.value.as_usize(),
                "lfm2.vocab_size" => vocab_size = entry.value.as_usize(),
                "lfm2.feed_forward_length" => intermediate_ffn_size = entry.value.as_usize(),
                "lfm2.rope.freq_base" => rope_theta = entry.value.as_f32(),
                "lfm2.context_length" => max_seq_len = entry.value.as_usize(),
                "lfm2.attention.layer_norm_rms_epsilon" => {
                    if let Some(v) = entry.value.as_f32() {
                        rms_epsilon = v;
                    }
                }
                "lfm2.layer_norm_rms_epsilon" => {
                    if let Some(v) = entry.value.as_f32() {
                        rms_epsilon = v;
                    }
                }
                "lfm2.shortconv.l_cache" => conv_l_cache = entry.value.as_usize(),
                _ => {}
            }
        }

        let n_heads = n_heads.ok_or_else(|| anyhow::anyhow!("Missing attention.head_count"))?;
        let n_layers = n_layers.ok_or_else(|| anyhow::anyhow!("Missing block_count"))?;
        let n_kv_heads = match n_kv_heads {
            Some(vec) if vec.len() == 1 => {
                let n_kv_heads_count = vec[0];
                if n_kv_heads_count == 1 {
                    vec![n_heads; n_layers]
                } else {
                    vec![n_kv_heads_count; n_layers]
                }  
            }
            Some(vec) if vec.len() == n_layers => vec,
            Some(vec) => {
                anyhow::bail!(
                    "KV heads length {} does not match layers {}",
                    vec.len(),
                    n_layers
                )
            }
            None => {
                vec![n_heads; n_layers]
            }
        };

        Ok(ModelConfig {
            file_path: PathBuf::from_str(gguf_info.headers.path.as_str())?,
            architecture,
            model_name: model_name.ok_or_else(|| anyhow::anyhow!("Missing general.name"))?,
            hidden_size: hidden_size.ok_or_else(|| anyhow::anyhow!("Missing embedding_length"))?,
            n_heads,
            n_kv_heads,
            n_layers,
            vocab_size: vocab_size.ok_or_else(|| anyhow::anyhow!("Missing vocab_size"))?,
            intermediate_ffn_size: intermediate_ffn_size
                .ok_or_else(|| anyhow::anyhow!("Missing feed_forward_length"))?,
            rope_theta: rope_theta.ok_or_else(|| anyhow::anyhow!("Missing rope.freq_base"))?,
            max_seq_len: max_seq_len.ok_or_else(|| anyhow::anyhow!("Missing context_length"))?,
            rms_epsilon,
            cache_seq_len_dim: Lfm2ExtraParameters::CACHE_SEQ_LEN_DIM,
            conv_l_cache,
            n_expert_used: None,
            n_expert_group: None,
            n_expert_group_used: None,
            expert_gating_func: None,
            n_leading_dense_block: None,
            q_lora_rank: None,
            kv_lora_rank: None,
            attn_key_length: None,
            attn_value_length: None,
            attn_key_length_mla: None,
            attn_value_length_mla: None,
            expert_ffn_size: None,
            n_experts: None,
            n_expert_shared: None,
            expert_weights_scale: None,
            expert_weights_norm: None,
            n_rope_dims: None,
            supports_reasoning: Lfm2ExtraParameters::SUPPORTS_REASONING,
            reasoning_start_token_id: Lfm2ExtraParameters::REASONING_START_TOKEN_ID,
            reasoning_end_token_id: Lfm2ExtraParameters::REASONING_END_TOKEN_ID,
            supports_tool_calling: Lfm2ExtraParameters::SUPPORTS_TOOL_CALLING,
            tool_call_start_token_id: Lfm2ExtraParameters::TOOL_CALL_START_TOKEN_ID,
            tool_call_end_token_id: Lfm2ExtraParameters::TOOL_CALL_END_TOKEN_ID,
            tool_call_format_style: Lfm2ExtraParameters::TOOL_CALL_FORMAT_STYLE,
            flatten_tools_to_functions: Lfm2ExtraParameters::FLATTEN_TOOLS_TO_FUNCTIONS,
            arg_key_open_token_id: None,
            arg_key_close_token_id: None,
            arg_value_open_token_id: None,
            arg_value_close_token_id: None,
        })
    }

    fn build_n_kv_count(gguf_meta_entry: &GgufKVMeta) -> anyhow::Result<Vec<usize>> {
        let result = match &gguf_meta_entry.value {
            GgufValue::Array(v) => v
                .iter()
                .map(|f| {
                    f.as_usize()
                        .ok_or_else(|| anyhow::anyhow!("Invalid KV head entry"))
                })
                .collect::<anyhow::Result<Vec<usize>>>()?,
            _ => vec![
                gguf_meta_entry
                    .value
                    .as_usize()
                    .ok_or_else(|| anyhow::anyhow!("Expected usize for KV count"))?,
            ],
        };

        Ok(result)
    }
}