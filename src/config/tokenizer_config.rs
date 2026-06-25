use crate::gguf::{GgufKVMeta, GgufValue};

pub struct TokenizerConfig {
    pub model_type: Option<String>, // from tokenizer.ggml.model
    pub pre_tokenizer_tag: Option<String>,
    pub tokens: Option<Vec<String>>,
    pub token_type: Option<Vec<i32>>,
    pub merges: Option<Vec<String>>,
    pub json_config: Option<String>,
    pub chat_template: Option<String>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub padding_token_id: Option<u32>,
    pub eot_token_id: Option<u32>,
    pub unknown_token_id: Option<u32>,
    pub eom_token_id: Option<u32>,
    pub add_bos: bool,
    pub add_eos: bool,
}

impl From<&[GgufKVMeta]> for TokenizerConfig {
    fn from(metadata: &[GgufKVMeta]) -> Self {
        let mut json_config = None;
        let mut model_type = None;
        let mut pre_tokenizer_tag = None;
        let mut tokens = None;
        let mut token_type = None;
        let mut merges = None;
        let mut chat_template = None;
        let mut bos_token_id = None;
        let mut eos_token_id = None;
        let mut padding_token_id = None;
        let mut eot_token_id = None;
        let mut unknown_token_id = None;
        let mut eom_token_id = None;
        let mut add_bos = false;
        let mut add_eos = false;

        for kv_meta in metadata {
            match kv_meta.key.as_str() {
                "tokenizer.ggml.hf_json" => {
                    json_config = kv_meta.value.as_string().map(|v| v.to_string())
                }
                "tokenizer.ggml.model" => {
                    model_type = kv_meta.value.as_string().map(|v| v.to_string())
                }
                "tokenizer.ggml.pre" => {
                    pre_tokenizer_tag = kv_meta.value.as_string().map(|v| v.to_string())
                }
                "tokenizer.ggml.tokens" => {
                    tokens = kv_meta.value.as_slice().and_then(|slice| {
                        slice
                            .iter()
                            .filter_map(|v: &GgufValue| v.as_string().map(|v| Some(v.to_string())))
                            .collect::<Option<Vec<String>>>()
                    })
                }
                "tokenizer.ggml.token_type" => {
                    token_type = kv_meta.value.as_slice().and_then(|slice| {
                        slice
                            .iter()
                            .filter_map(|v: &GgufValue| v.as_i32().map(Some))
                            .collect::<Option<Vec<i32>>>()
                    })
                }
                "tokenizer.ggml.merges" => {
                    merges = kv_meta.value.as_slice().and_then(|slice| {
                        slice
                            .iter()
                            .filter_map(|v: &GgufValue| v.as_string().map(|v| Some(v.to_string())))
                            .collect::<Option<Vec<String>>>()
                    })
                }
                "tokenizer.chat_template" => {
                    chat_template = kv_meta.value.as_string().map(|v| v.to_string())
                }
                "tokenizer.ggml.bos_token_id" => bos_token_id = kv_meta.value.as_u32(),
                "tokenizer.ggml.eos_token_id" => eos_token_id = kv_meta.value.as_u32(),
                "tokenizer.ggml.padding_token_id" => padding_token_id = kv_meta.value.as_u32(),
                "tokenizer.ggml.eot_token_id" => eot_token_id = kv_meta.value.as_u32(),
                "tokenizer.ggml.unknown_token_id" => unknown_token_id = kv_meta.value.as_u32(),
                "tokenizer.ggml.eom_token_id" => eom_token_id = kv_meta.value.as_u32(),
                "tokenizer.ggml.add_bos_token" => {
                    add_bos = kv_meta.value.as_bool().is_some_and(|v| v)
                }
                "tokenizer.ggml.add_eos_token" => {
                    add_eos = kv_meta.value.as_bool().is_some_and(|v| v)
                }
                _ => {}
            }
        }

        Self {
            model_type,
            pre_tokenizer_tag,
            tokens,
            token_type,
            merges,
            json_config,
            chat_template,
            bos_token_id,
            eos_token_id,
            padding_token_id,
            eot_token_id,
            unknown_token_id,
            eom_token_id,
            add_bos,
            add_eos,
        }
    }
}
