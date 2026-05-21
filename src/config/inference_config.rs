use candle_core::DType;

pub struct InferenceConfig {
    pub dtype: DType,
    pub max_seq_len: usize,
    pub conv_on_cpu: bool
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl InferenceConfig {
    pub fn builder() -> InferenceConfigBuilder {
        InferenceConfigBuilder::new()
    }
}

pub struct InferenceConfigBuilder {
    pub dtype: DType,
    pub max_seq_len: usize,
    pub conv_on_cpu: bool,
}

impl InferenceConfigBuilder {
    pub fn new() -> Self {
        Self {
            dtype: DType::F16,
            max_seq_len: 32_000,
            conv_on_cpu: True,
        }
    }

    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    pub fn conv_on_cpu(mut self, conv_on_cpu: bool) -> Self {
        self.conv_on_cpu = conv_on_cpu;
        self
    }

    pub fn build(self) -> InferenceConfig {
        InferenceConfig {
            dtype: self.dtype,
            max_seq_len: self.max_seq_len,
            conv_on_cpu: self.conv_on_cpu,
        }
    }
}