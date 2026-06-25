use crate::gguf::{GgufKVMeta, GgufValue};

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
