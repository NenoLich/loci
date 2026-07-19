use std::path::PathBuf;
use std::str::FromStr;

use crate::config::parser::build_n_kv_count;
use crate::config::{ModelArchitecture, ModelConfig};
use crate::gguf::GgufInfo;
use crate::inference::ToolFormatStyle;

pub struct DeepSeek2ExtraParameters;

impl DeepSeek2ExtraParameters {
    pub const CACHE_SEQ_LEN_DIM: usize = 2;
    pub const SUPPORTS_TOOL_CALLING: bool = true;
    pub const SUPPORTS_REASONING: bool = true;
    pub const TOOL_CALL_START_TOKEN_ID: Option<u32> = Some(151352);
    pub const TOOL_CALL_END_TOKEN_ID: Option<u32> = Some(151353);
    pub const REASONING_START_TOKEN_ID: Option<u32> = Some(151350);
    pub const REASONING_END_TOKEN_ID: Option<u32> = Some(151351);
    pub const TOOL_CALL_FORMAT_STYLE: ToolFormatStyle = ToolFormatStyle::XmlArgPairs;
    pub const ARG_KEY_OPEN_TOKEN_ID: Option<u32> = Some(151356);
    pub const ARG_KEY_CLOSE_TOKEN_ID: Option<u32> = Some(151357);
    pub const ARG_VALUE_OPEN_TOKEN_ID: Option<u32> = Some(151358);
    pub const ARG_VALUE_CLOSE_TOKEN_ID: Option<u32> = Some(151359);
}

pub struct Deepseek2Parser;

impl Deepseek2Parser {
    pub fn parse(gguf_info: &GgufInfo) -> anyhow::Result<ModelConfig> {
        let architecture = ModelArchitecture::Deepseek2;
        let mut model_name = None;
        let mut n_layers = None;
        let mut max_seq_len = None;
        let mut hidden_size = None;
        let mut intermediate_ffn_size = None;
        let mut n_heads = None;
        let mut n_kv_heads = None;
        let mut rope_theta = None;
        let mut rms_epsilon = 1e-5;
        let mut n_expert_used = None;
        let mut n_expert_group = None;
        let mut n_expert_group_used = None;
        let mut expert_gating_func = None;
        let mut n_leading_dense_block = None;
        let mut vocab_size = None;
        let mut q_lora_rank = None;
        let mut kv_lora_rank = None;
        let mut attn_key_length = None;
        let mut attn_value_length = None;
        let mut attn_key_length_mla = None;
        let mut attn_value_length_mla = None;
        let mut expert_ffn_size = None;
        let mut n_experts = None;
        let mut n_expert_shared = None;
        let mut expert_weights_scale = None;
        let mut expert_weights_norm = None;
        let mut n_rope_dims = None;

        let metadata = &gguf_info.kv_meta;
        for entry in metadata {
            match entry.key.as_str() {
                "general.name" => model_name = entry.value.as_str(),
                "deepseek2.block_count" => n_layers = entry.value.as_usize(),
                "deepseek2.context_length" => max_seq_len = entry.value.as_usize(),
                "deepseek2.embedding_length" => hidden_size = entry.value.as_usize(),
                "deepseek2.feed_forward_length" => intermediate_ffn_size = entry.value.as_usize(),
                "deepseek2.attention.head_count" => n_heads = entry.value.as_usize(),
                "deepseek2.attention.head_count_kv" => n_kv_heads = Some(build_n_kv_count(entry)?),
                "deepseek2.rope.freq_base" => rope_theta = entry.value.as_f32(),
                "deepseek2.attention.layer_norm_rms_epsilon" => {
                    if let Some(value) = entry.value.as_f32() {
                        rms_epsilon = value;
                    }
                }
                "deepseek2.expert_used_count" => n_expert_used = entry.value.as_usize(),
                "deepseek2.expert_group_count" => n_expert_group = entry.value.as_usize(),
                "deepseek2.expert_group_used_count" => n_expert_group_used = entry.value.as_usize(),
                "deepseek2.expert_gating_func" => expert_gating_func = entry.value.as_usize(),
                "deepseek2.leading_dense_block_count" => {
                    n_leading_dense_block = entry.value.as_usize()
                }
                "deepseek2.vocab_size" => vocab_size = entry.value.as_usize(),
                "deepseek2.attention.q_lora_rank" => q_lora_rank = entry.value.as_usize(),
                "deepseek2.attention.kv_lora_rank" => kv_lora_rank = entry.value.as_usize(),
                "deepseek2.attention.key_length" => attn_key_length = entry.value.as_usize(),
                "deepseek2.attention.value_length" => attn_value_length = entry.value.as_usize(),
                "deepseek2.attention.key_length_mla" => {
                    attn_key_length_mla = entry.value.as_usize()
                }
                "deepseek2.attention.value_length_mla" => {
                    attn_value_length_mla = entry.value.as_usize()
                }
                "deepseek2.expert_feed_forward_length" => expert_ffn_size = entry.value.as_usize(),
                "deepseek2.expert_count" => n_experts = entry.value.as_usize(),
                "deepseek2.expert_shared_count" => n_expert_shared = entry.value.as_usize(),
                "deepseek2.expert_weights_scale" => expert_weights_scale = entry.value.as_f32(),
                "deepseek2.expert_weights_norm" => expert_weights_norm = entry.value.as_bool(),
                "deepseek2.rope.dimension_count" => n_rope_dims = entry.value.as_usize(),
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
            model_name: model_name
                .ok_or_else(|| anyhow::anyhow!("Missing general.name"))?
                .to_string(),
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
            cache_seq_len_dim: DeepSeek2ExtraParameters::CACHE_SEQ_LEN_DIM,
            conv_l_cache: None,
            n_expert_used,
            n_expert_group,
            n_expert_group_used,
            expert_gating_func,
            n_leading_dense_block,
            q_lora_rank,
            kv_lora_rank,
            attn_key_length,
            attn_value_length,
            attn_key_length_mla,
            attn_value_length_mla,
            expert_ffn_size,
            n_experts,
            n_expert_shared,
            expert_weights_scale,
            expert_weights_norm,
            n_rope_dims,
            supports_reasoning: DeepSeek2ExtraParameters::SUPPORTS_REASONING,
            supports_tool_calling: DeepSeek2ExtraParameters::SUPPORTS_TOOL_CALLING,
            tool_call_start_token_id: DeepSeek2ExtraParameters::TOOL_CALL_START_TOKEN_ID,
            tool_call_end_token_id: DeepSeek2ExtraParameters::TOOL_CALL_END_TOKEN_ID,
            reasoning_start_token_id: DeepSeek2ExtraParameters::REASONING_START_TOKEN_ID,
            reasoning_end_token_id: DeepSeek2ExtraParameters::REASONING_END_TOKEN_ID,
            tool_call_format_style: DeepSeek2ExtraParameters::TOOL_CALL_FORMAT_STYLE,
            flatten_tools_to_functions: false,
            arg_key_open_token_id: DeepSeek2ExtraParameters::ARG_KEY_OPEN_TOKEN_ID,
            arg_key_close_token_id: DeepSeek2ExtraParameters::ARG_KEY_CLOSE_TOKEN_ID,
            arg_value_open_token_id: DeepSeek2ExtraParameters::ARG_VALUE_OPEN_TOKEN_ID,
            arg_value_close_token_id: DeepSeek2ExtraParameters::ARG_VALUE_CLOSE_TOKEN_ID,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufHeaders, GgufInfo, GgufKVMeta, GgufTensorInfo, GgufType, GgufValue};
    use rstest::rstest;

    fn setup_test_gguf_info() -> GgufInfo {
        GgufInfo {
            headers: GgufHeaders {
                path: "test_path".to_string(),
                magic: "GGUF".to_string(),
                version: 5342,
                tensor_count: 1,
                metadata_kv_count: 6,
            },
            kv_meta: vec![
                GgufKVMeta {
                    key: "general.name".to_string(),
                    value_type: GgufType::String,
                    value: GgufValue::String("test_name".to_string()),
                },
                GgufKVMeta {
                    key: "deepseek2.embedding_length".to_string(),
                    value_type: GgufType::Uint32,
                    value: GgufValue::Uint32(512),
                },
                GgufKVMeta {
                    key: "deepseek2.feed_forward_length".to_string(),
                    value_type: GgufType::Uint32,
                    value: GgufValue::Uint32(2048),
                },
                GgufKVMeta {
                    key: "deepseek2.context_length".to_string(),
                    value_type: GgufType::Uint32,
                    value: GgufValue::Uint32(2048),
                },
                GgufKVMeta {
                    key: "deepseek2.vocab_size".to_string(),
                    value_type: GgufType::Uint32,
                    value: GgufValue::Uint32(32000),
                },
                GgufKVMeta {
                    key: "deepseek2.rope.freq_base".to_string(),
                    value_type: GgufType::Float32,
                    value: GgufValue::Float32(10000.0),
                },
                GgufKVMeta {
                    key: "deepseek2.attention.head_count".to_string(),
                    value_type: GgufType::Uint32,
                    value: GgufValue::Uint32(12),
                },
                GgufKVMeta {
                    key: "deepseek2.block_count".to_string(),
                    value_type: GgufType::Uint32,
                    value: GgufValue::Uint32(2),
                },
                GgufKVMeta {
                    key: "deepseek2.attention.head_count_kv".to_string(),
                    value_type: GgufType::Array,
                    value: GgufValue::Array(vec![GgufValue::Uint32(12), GgufValue::Uint32(12)]),
                },
            ],
            tensor_info: vec![GgufTensorInfo {
                name: "test_tensor_name".to_string(),
                n_dims: 3,
                shapes: vec![1, 2, 3],
                ggml_type: 3,
                offset: 0,
            }],
            tensor_offset_start: 0,
        }
    }

    fn setup_test_model_config() -> ModelConfig {
        ModelConfig {
            file_path: PathBuf::from("test_path"),
            architecture: ModelArchitecture::Deepseek2,
            model_name: "test_name".to_string(),
            hidden_size: 512,
            n_heads: 12,
            n_kv_heads: vec![12, 12],
            n_layers: 2,
            vocab_size: 32000,
            intermediate_ffn_size: 2048,
            rope_theta: 10000.0,
            max_seq_len: 2048,
            rms_epsilon: 1e-5,
            cache_seq_len_dim: DeepSeek2ExtraParameters::CACHE_SEQ_LEN_DIM,
            supports_reasoning: DeepSeek2ExtraParameters::SUPPORTS_REASONING,
            supports_tool_calling: DeepSeek2ExtraParameters::SUPPORTS_TOOL_CALLING,
            tool_call_start_token_id: DeepSeek2ExtraParameters::TOOL_CALL_START_TOKEN_ID,
            tool_call_end_token_id: DeepSeek2ExtraParameters::TOOL_CALL_END_TOKEN_ID,
            reasoning_start_token_id: DeepSeek2ExtraParameters::REASONING_START_TOKEN_ID,
            reasoning_end_token_id: DeepSeek2ExtraParameters::REASONING_END_TOKEN_ID,
            tool_call_format_style: DeepSeek2ExtraParameters::TOOL_CALL_FORMAT_STYLE,
            flatten_tools_to_functions: false,
            arg_key_open_token_id: DeepSeek2ExtraParameters::ARG_KEY_OPEN_TOKEN_ID,
            arg_key_close_token_id: DeepSeek2ExtraParameters::ARG_KEY_CLOSE_TOKEN_ID,
            arg_value_open_token_id: DeepSeek2ExtraParameters::ARG_VALUE_OPEN_TOKEN_ID,
            arg_value_close_token_id: DeepSeek2ExtraParameters::ARG_VALUE_CLOSE_TOKEN_ID,
            ..Default::default()
        }
    }

    #[rstest]
    #[case({
        let mut gguf_info = setup_test_gguf_info();
        gguf_info.kv_meta.retain(|entry| entry.key != "deepseek2.attention.head_count".to_string());
        gguf_info
    },
    "Missing attention.head_count")]
    #[case({
        let mut gguf_info = setup_test_gguf_info();
        gguf_info.kv_meta.retain(|entry| entry.key != "deepseek2.block_count".to_string());
        gguf_info
    },
    "Missing block_count")]
    #[case({
        let mut gguf_info = setup_test_gguf_info();
        let n_kv_heads = gguf_info.kv_meta.iter_mut().find(|entry| entry.key == "deepseek2.attention.head_count_kv".to_string()).unwrap();
        n_kv_heads.value = GgufValue::Array(vec![GgufValue::Uint32(12), GgufValue::Uint32(12), GgufValue::Uint32(12)]);
        gguf_info
    },
    "KV heads length 3 does not match layers 2")]
    #[case({
        let mut gguf_info = setup_test_gguf_info();
        gguf_info.kv_meta.retain(|entry| entry.key != "general.name".to_string());
        gguf_info
    },
    "Missing general.name")]
    #[case({
        let mut gguf_info = setup_test_gguf_info();
        gguf_info.kv_meta.retain(|entry| entry.key != "deepseek2.embedding_length".to_string());
        gguf_info
    },
    "Missing embedding_length")]
    #[case({
        let mut gguf_info = setup_test_gguf_info();
        gguf_info.kv_meta.retain(|entry| entry.key != "deepseek2.feed_forward_length".to_string());
        gguf_info
    },
    "Missing feed_forward_length")]
    #[case({
        let mut gguf_info = setup_test_gguf_info();
        gguf_info.kv_meta.retain(|entry| entry.key != "deepseek2.context_length".to_string());
        gguf_info
    },
    "Missing context_length")]
    #[case({
        let mut gguf_info = setup_test_gguf_info();
        gguf_info.kv_meta.retain(|entry| entry.key != "deepseek2.vocab_size".to_string());
        gguf_info
    },
    "Missing vocab_size")]
    #[case({
        let mut gguf_info = setup_test_gguf_info();
        gguf_info.kv_meta.retain(|entry| entry.key != "deepseek2.rope.freq_base".to_string());
        gguf_info
    },
    "Missing rope.freq_base")]
    fn test_deepseek2_parse_failure(#[case] gguf_info: GgufInfo, #[case] expected_error: &str) {
        let result = Deepseek2Parser::parse(&gguf_info).expect_err("expected parse to fail");
        assert!(result.to_string().contains(expected_error));
    }

    #[rstest]
    #[case(setup_test_gguf_info(), setup_test_model_config())]
    #[case({
        let mut gguf_info = setup_test_gguf_info();
        gguf_info.kv_meta.push(GgufKVMeta {
            key: "deepseek2.attention.layer_norm_rms_epsilon".to_string(),
            value_type: GgufType::Float32,
            value: GgufValue::Float32(1e-3),
        });
        gguf_info
    },
    {
        let mut config = setup_test_model_config();
        config.rms_epsilon = 1e-3;
        config
    })]
    #[case({
        let mut gguf_info = setup_test_gguf_info();
        let n_kv_heads = gguf_info.kv_meta.iter_mut().find(|entry| entry.key == "deepseek2.attention.head_count_kv".to_string()).unwrap();
        n_kv_heads.value = GgufValue::Array(vec![GgufValue::Uint32(1)]);
        gguf_info
    },
    {
        let mut config = setup_test_model_config();
        config.n_kv_heads = vec![config.n_heads; config.n_layers];
        config
    })]
    #[case({
        let mut gguf_info = setup_test_gguf_info();
        let n_kv_heads = gguf_info.kv_meta.iter_mut().find(|entry| entry.key == "deepseek2.attention.head_count_kv".to_string()).unwrap();
        n_kv_heads.value = GgufValue::Array(vec![GgufValue::Uint32(12)]);
        gguf_info
    },
    {
        let mut config = setup_test_model_config();
        config.n_kv_heads = vec![12; config.n_layers];
        config
    })]
    #[case({
        let mut gguf_info = setup_test_gguf_info();
        gguf_info.kv_meta.retain(|entry| entry.key != "deepseek2.attention.head_count_kv".to_string());
        gguf_info
    },
    {
        let mut config = setup_test_model_config();
        config.n_kv_heads = vec![config.n_heads; config.n_layers];
        config
    })]
    fn test_deepseek2_parse_success(
        #[case] gguf_info: GgufInfo,
        #[case] expected_config: ModelConfig,
    ) {
        let result = Deepseek2Parser::parse(&gguf_info).expect("expected parse to succeed");
        assert_eq!(result, expected_config);
    }
}
