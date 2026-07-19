use crate::gguf::{GgufKVMeta, GgufValue};

/// Build a vector of KV count from a GgufKVMeta entry
///
/// # Examples
/// ```
/// use loci::gguf::{GgufKVMeta, GgufValue, GgufType};
/// use loci::config::parser::build_n_kv_count;
///
/// let gguf_meta_entry_array = GgufKVMeta {
///     key: String::from("lfm2.attention.head_count_kv"),
///     value_type: GgufType::Array,
///     value: GgufValue::Array(vec![GgufValue::Uint32(1), GgufValue::Uint32(2)]),
/// };
/// let result = build_n_kv_count(&gguf_meta_entry_array);
/// assert!(result.is_ok());
/// let result = result.unwrap();
/// assert_eq!(result, vec![1, 2]);
///
/// let gguf_meta_entry_usize = GgufKVMeta {
///     key: String::from("lfm2.attention.head_count_kv"),
///     value_type: GgufType::Uint32,
///     value: GgufValue::Uint32(2),
/// };
/// let result = build_n_kv_count(&gguf_meta_entry_usize);
/// assert!(result.is_ok());
/// let result = result.unwrap();
/// assert_eq!(result, vec![2]);
/// ```
pub fn build_n_kv_count(gguf_meta_entry: &GgufKVMeta) -> anyhow::Result<Vec<usize>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufType;
    use rstest::rstest;

    #[rstest]
    #[case(GgufKVMeta {
        key: String::from("lfm2.attention.head_count_kv"),
        value_type: GgufType::Float32,
        value: GgufValue::Float32(2.0),
    }, "Expected usize for KV count")]
    #[case(GgufKVMeta {
        key: String::from("lfm2.attention.head_count_kv"),
        value_type: GgufType::Array,
        value: GgufValue::Array(vec![GgufValue::Float32(2.0), GgufValue::Float32(4.0)]),
    }, "Invalid KV head entry")]
    fn test_build_n_kv_count_failure(
        #[case] gguf_meta_entry: GgufKVMeta,
        #[case] expected_error_message_str: &str,
    ) {
        let result = build_n_kv_count(&gguf_meta_entry)
            .expect_err("build_n_kv_count should fail, but did not");
        assert!(result.to_string().contains(expected_error_message_str));
    }

    #[rstest]
    #[case(GgufKVMeta {
        key: String::from("lfm2.attention.head_count_kv"),
        value_type: GgufType::Array,
        value: GgufValue::Array(vec![]),
    }, vec![])]
    #[case(GgufKVMeta {
        key: String::from("lfm2.attention.head_count_kv"),
        value_type: GgufType::Array,
        value: GgufValue::Array(vec![GgufValue::Uint32(1), GgufValue::Uint32(2)]),
    }, vec![1, 2])]
    #[case(GgufKVMeta {
        key: String::from("lfm2.attention.head_count_kv"),
        value_type: GgufType::Uint32,
        value: GgufValue::Uint32(2),
    }, vec![2])]
    fn test_build_n_kv_count_success(
        #[case] gguf_meta_entry: GgufKVMeta,
        #[case] expected_result: Vec<usize>,
    ) {
        let result = build_n_kv_count(&gguf_meta_entry)
            .expect("build_n_kv_count should succeed, but did not");
        assert_eq!(result, expected_result);
    }
}
