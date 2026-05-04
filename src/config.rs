use anyhow::{Error, Ok, Result, bail};
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

use crate::gguf::{GgufInfo, GgufKVMeta, GgufValue};

#[derive(Debug, Clone)]
pub enum ModelArchitecture {
    Lfm2,
    Llama,
}

impl FromStr for ModelArchitecture {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "lfm2" => Ok(ModelArchitecture::Lfm2),
            "llama" => Ok(ModelArchitecture::Llama),
            _ => bail!("Unsupported model architecture: {}", s),
        }
    }
}

impl fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ModelArchitecture::Lfm2 => write!(f, "Lfm2"),
            ModelArchitecture::Llama => write!(f, "Llama"),
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
    pub conv_l_cache: usize,
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
            ModelArchitecture::Lfm2 => Self::parse_lfm2(gguf_info),
            _ => bail!("Unsupported model architecture: {}", arch),
        }
    }

    fn parse_lfm2(gguf_info: &GgufInfo) -> Result<Self> {
        let mut hidden_size = None;
        let mut n_heads = None;
        let mut n_kv_heads = None;
        let mut n_layers = None;
        let mut vocab_size = None;
        let mut intermediate_ffn_size = None;
        let mut rope_theta = None;
        let mut max_seq_len = None;
        let mut rms_epsilon = 1e-5;
        let mut conv_l_cache = 3;

        let metadata = &gguf_info.kv_meta;
        for entry in metadata {
            match entry.key.as_str() {
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
                "lfm2.shortconv.l_cache" => {
                    if let Some(v) = entry.value.as_usize() {
                        conv_l_cache = v;
                    }
                }
                _ => {}
            }
        }

        let n_heads = n_heads.ok_or_else(|| anyhow::anyhow!("Missing attention.head_count"))?;
        let n_layers = n_layers.ok_or_else(|| anyhow::anyhow!("Missing block_count"))?;
        let n_kv_heads = match n_kv_heads {
            Some(vec) if vec.len() == 1 => {
                vec![vec[0]; n_layers]
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
                vec![n_heads, n_layers]
            }
        };

        Ok(Self {
            file_path: PathBuf::from_str(gguf_info.headers.path.as_str())?,
            architecture: ModelArchitecture::Lfm2,
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
            conv_l_cache,
        })
    }

    fn build_n_kv_count(gguf_meta_entry: &GgufKVMeta) -> Result<Vec<usize>> {
        let result = match &gguf_meta_entry.value {
            GgufValue::Array(v) => v
                .iter()
                .map(|f| {
                    f.as_usize()
                        .ok_or_else(|| anyhow::anyhow!("Invalid KV head entry"))
                })
                .collect::<Result<Vec<usize>>>()?,
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
