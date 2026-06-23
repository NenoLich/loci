use candle_core::DType;
use crate::config::{ComputeDtype, FileConfig, InferenceFileConfig};

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub dtype: DType,
    pub max_seq_len: usize,
    pub flash_attn: bool,
    pub conv_on_cpu: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            dtype: DType::F16,
            max_seq_len: 32_000,
            flash_attn: true,
            conv_on_cpu: true 
        }
    }
}

impl InferenceConfig {
    pub fn builder() -> InferenceConfigBuilder {
        InferenceConfigBuilder::default()
    }
}

#[derive(Default)]
pub struct InferenceConfigBuilder {
    pub dtype: Option<DType>,
    pub max_seq_len: Option<usize>,
    pub flash_attn: Option<bool>,
    pub conv_on_cpu: Option<bool>,
    pub file_config: Option<InferenceFileConfig>
}

impl InferenceConfigBuilder {
    pub fn dtype(mut self, compute_dtype: Option<ComputeDtype>) -> Self {
        if let Some(dtype) = compute_dtype {
            self.dtype = match dtype {
                ComputeDtype::F16 => Some(DType::F16),
                ComputeDtype::F32 => Some(DType::F32),
            }
        };
        self
        
    }

    pub fn max_seq_len(mut self, max_seq_len: Option<usize>) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    pub fn flash_attn(mut self, flash_attn: Option<bool>) -> Self {
        self.flash_attn = flash_attn;
        self
    }

    pub fn conv_on_cpu(mut self, conv_on_cpu: Option<bool>) -> Self {
        self.conv_on_cpu = conv_on_cpu;
        self
    }

    pub fn with_file_config(mut self, config: Option<InferenceFileConfig>) -> Self {
        self.file_config = config;
        self
    }

    pub fn build(self) -> InferenceConfig {
        let default = InferenceConfig::default();
        InferenceConfig {
            dtype: self.dtype
                .or_else(|| self.file_config.as_ref().and_then(|c| 
                    match c.dtype.as_ref() {
                        Some(ComputeDtype::F16) => Some(DType::F16),
                        Some(ComputeDtype::F32) => Some(DType::F32),
                        _ => None,
                }))
                .unwrap_or(default.dtype),
            max_seq_len: self.max_seq_len
                .or_else(|| self.file_config.as_ref().and_then(|c| c.max_seq_len))
                .unwrap_or(default.max_seq_len),
            flash_attn: self.flash_attn
                .or_else(|| self.file_config.as_ref().and_then(|c| c.flash_attn))
                .unwrap_or(default.flash_attn),
            conv_on_cpu: self.conv_on_cpu
                .or_else(|| self.file_config.as_ref().and_then(|c| c.conv_on_cpu))
                .unwrap_or(default.conv_on_cpu),
        }
    }
}