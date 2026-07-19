use crate::config::GenerationFileConfig;
use crate::gguf::GgufInfo;
use crate::types::{ReasoningEffort, ToolChoice, ToolChoiceMode};

/// Generation parameters with priority: CLI > GGUF metadata > defaults
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub tool_choice: ToolChoice,
    pub reasoning_effort: ReasoningEffort,
    pub stop_tokens: Option<Vec<String>>,
    pub logprobs: bool,
    pub top_logprobs: Option<usize>,
    pub seed: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 32_000,
            temperature: 0.8,
            top_p: 0.9,
            repetition_penalty: 1.15,
            tool_choice: ToolChoice::Mode(ToolChoiceMode::Auto),
            reasoning_effort: ReasoningEffort::High,
            stop_tokens: None,
            logprobs: false,
            top_logprobs: None,
            seed: 19,
        }
    }
}

impl GenerationConfig {
    /// Create a new builder for generation config
    pub fn builder() -> GenerationConfigBuilder {
        GenerationConfigBuilder::default()
    }

    /// Resolve from GGUF metadata
    pub fn from_gguf_metadata(gguf_info: &GgufInfo) -> anyhow::Result<Self> {
        let default = GenerationConfig::default();
        let mut temperature = default.temperature;
        let mut max_tokens = default.max_tokens;
        let mut top_p = default.top_p;
        let mut repetition_penalty = default.repetition_penalty;
        let tool_choice = default.tool_choice;
        let reasoning_effort = default.reasoning_effort;
        let stop_tokens = default.stop_tokens;
        let logprobs = default.logprobs;
        let top_logprobs = default.top_logprobs;
        let seed = default.seed;

        let metadata = &gguf_info.kv_meta;

        let _architecture = metadata
            .iter()
            .find(|entry| entry.key == "general.architecture")
            .and_then(|entry| entry.value.as_str())
            .ok_or_else(|| {
                anyhow::anyhow!("Could not find 'general.architecture' key in gguf metadata")
            })?;

        for entry in metadata {
            match entry.key.as_str() {
                "sampling.temperature"
                | "general.sampling.temp"
                | "general.sampling.temperature" => {
                    if let Some(value) = entry.value.as_f32() {
                        temperature = value;
                    }
                }
                "sampling.top_p" | "general.sampling.top_p" => {
                    if let Some(value) = entry.value.as_f32() {
                        top_p = value;
                    }
                }
                "sampling.repetition_penalty" | "general.sampling.repetition_penalty" => {
                    if let Some(value) = entry.value.as_f32() {
                        repetition_penalty = value;
                    }
                }
                "general.max_tokens" => {
                    if let Some(value) = entry.value.as_usize() {
                        max_tokens = value;
                    }
                }
                _ => {}
            }
        }

        Ok(Self {
            max_tokens,
            temperature,
            top_p,
            repetition_penalty,
            tool_choice,
            reasoning_effort,
            stop_tokens,
            logprobs,
            top_logprobs,
            seed,
        })
    }
}

/// Builder for GenerationConfig with priority resolution
#[derive(Debug, Default, Clone)]
pub struct GenerationConfigBuilder {
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<usize>,
    repetition_penalty: Option<f32>,
    tool_choice: Option<ToolChoice>,
    reasoning_effort: Option<ReasoningEffort>,
    stop_tokens: Option<Vec<String>>,
    logprobs: Option<bool>,
    top_logprobs: Option<usize>,
    seed: Option<usize>,
    gguf_config: Option<GenerationConfig>,
    file_config: Option<GenerationFileConfig>,
}

impl GenerationConfigBuilder {
    /// Set explicit temperature (CLI argument)
    #[allow(dead_code)]
    pub fn temperature(mut self, temp: Option<f32>) -> Self {
        self.temperature = temp;
        self
    }

    /// Set explicit top_p (CLI argument)
    #[allow(dead_code)]
    pub fn top_p(mut self, top_p: Option<f32>) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set explicit max_tokens (CLI argument)
    #[allow(dead_code)]
    pub fn max_tokens(mut self, tokens: Option<usize>) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set explicit repetition_penalty (CLI argument)
    #[allow(dead_code)]
    pub fn repetition_penalty(mut self, penalty: Option<f32>) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Set explicit tool_choice (CLI argument)
    #[allow(dead_code)]
    pub fn tool_choice(mut self, tool_choice: Option<ToolChoice>) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    /// Set explicit reasoning_effort (CLI argument)
    #[allow(dead_code)]
    pub fn reasoning_effort(mut self, effort: Option<ReasoningEffort>) -> Self {
        self.reasoning_effort = effort;
        self
    }

    /// Set explicit stop_tokens (CLI argument)
    #[allow(dead_code)]
    pub fn stop_tokens(mut self, stop_tokens: Option<Vec<String>>) -> Self {
        self.stop_tokens = stop_tokens;
        self
    }

    /// Set explicit logprobs (CLI argument)
    #[allow(dead_code)]
    pub fn logprobs(mut self, logprobs: Option<bool>) -> Self {
        self.logprobs = logprobs;
        self
    }

    /// Set explicit top_logprobs (CLI argument)
    #[allow(dead_code)]
    pub fn top_logprobs(mut self, top_logprobs: Option<usize>) -> Self {
        self.top_logprobs = top_logprobs;
        self
    }

    /// Set explicit seed (CLI argument)
    #[allow(dead_code)]
    pub fn seed(mut self, seed: Option<usize>) -> Self {
        self.seed = seed;
        self
    }

    /// Load from GGUF metadata (fallback source)
    pub fn with_gguf_metadata(mut self, gguf_info: &GgufInfo) -> anyhow::Result<Self> {
        self.gguf_config = Some(GenerationConfig::from_gguf_metadata(gguf_info)?);
        Ok(self)
    }

    pub fn with_overrides(mut self, overrides: GenerationOverrides) -> Self {
        self.temperature = if let Some(temp) = overrides.temperature {
            Some(temp)
        } else {
            self.temperature
        };
        self.top_p = if let Some(top_p) = overrides.top_p {
            Some(top_p)
        } else {
            self.top_p
        };
        self.max_tokens = if let Some(tokens) = overrides.max_tokens {
            Some(tokens)
        } else {
            self.max_tokens
        };
        self.repetition_penalty = if let Some(penalty) = overrides.repetition_penalty {
            Some(penalty)
        } else {
            self.repetition_penalty
        };
        self.tool_choice = if let Some(tool_choice) = overrides.tool_choice {
            Some(tool_choice)
        } else {
            self.tool_choice
        };
        self.reasoning_effort = if let Some(effort) = overrides.reasoning_effort {
            Some(effort)
        } else {
            self.reasoning_effort
        };
        self.stop_tokens = if let Some(stop_tokens) = overrides.stop_tokens {
            Some(stop_tokens)
        } else {
            self.stop_tokens
        };
        self.logprobs = if let Some(logprobs) = overrides.logprobs {
            Some(logprobs)
        } else {
            self.logprobs
        };
        self.top_logprobs = if let Some(top_logprobs) = overrides.top_logprobs {
            Some(top_logprobs)
        } else {
            self.top_logprobs
        };
        self.seed = if let Some(seed) = overrides.seed {
            Some(seed)
        } else {
            self.seed
        };
        self.file_config = if let Some(file_config) = overrides.file_config {
            Some(file_config)
        } else {
            self.file_config
        };
        self
    }

    #[allow(dead_code)]
    pub fn with_file_config(mut self, config: Option<GenerationFileConfig>) -> Self {
        self.file_config = config;
        self
    }

    /// Resolve with priority: explicit > gguf > defaults
    pub fn build(self) -> GenerationConfig {
        let default = GenerationConfig::default();
        GenerationConfig {
            temperature: self
                .temperature
                .or_else(|| self.file_config.as_ref().and_then(|c| c.temperature))
                .or_else(|| self.gguf_config.as_ref().map(|c| c.temperature))
                .unwrap_or(default.temperature),
            top_p: self
                .top_p
                .or_else(|| self.file_config.as_ref().and_then(|c| c.top_p))
                .or_else(|| self.gguf_config.as_ref().map(|c| c.top_p))
                .unwrap_or(default.top_p),
            max_tokens: self
                .max_tokens
                .or_else(|| self.file_config.as_ref().and_then(|c| c.max_tokens))
                .or_else(|| self.gguf_config.as_ref().map(|c| c.max_tokens))
                .unwrap_or(default.max_tokens),
            repetition_penalty: self
                .repetition_penalty
                .or_else(|| self.file_config.as_ref().and_then(|c| c.repetition_penalty))
                .or_else(|| self.gguf_config.as_ref().map(|c| c.repetition_penalty))
                .unwrap_or(default.repetition_penalty),
            tool_choice: self
                .tool_choice
                .or_else(|| {
                    self.file_config
                        .as_ref()
                        .and_then(|c| c.tool_choice.clone())
                })
                .or_else(|| self.gguf_config.as_ref().map(|c| c.tool_choice.clone()))
                .unwrap_or(default.tool_choice),
            reasoning_effort: self
                .reasoning_effort
                .or_else(|| {
                    self.file_config
                        .as_ref()
                        .and_then(|c| c.reasoning_effort.clone())
                })
                .or_else(|| {
                    self.gguf_config
                        .as_ref()
                        .map(|c| c.reasoning_effort.clone())
                })
                .unwrap_or(default.reasoning_effort),
            stop_tokens: self
                .stop_tokens
                .or_else(|| {
                    self.file_config
                        .as_ref()
                        .and_then(|c| c.stop_tokens.clone())
                })
                .or_else(|| {
                    self.gguf_config
                        .as_ref()
                        .and_then(|c| c.stop_tokens.clone())
                }),
            logprobs: self
                .logprobs
                .or_else(|| self.file_config.as_ref().and_then(|c| c.logprobs))
                .or_else(|| self.gguf_config.as_ref().map(|c| c.logprobs))
                .unwrap_or(default.logprobs),
            top_logprobs: self
                .top_logprobs
                .or_else(|| self.file_config.as_ref().and_then(|c| c.top_logprobs))
                .or_else(|| self.gguf_config.as_ref().and_then(|c| c.top_logprobs)),
            seed: self
                .seed
                .or_else(|| self.file_config.as_ref().and_then(|c| c.seed))
                .unwrap_or(default.seed),
        }
    }
}

#[derive(Default, Clone)]
pub struct GenerationOverrides {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub tool_choice: Option<ToolChoice>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub stop_tokens: Option<Vec<String>>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<usize>,
    pub seed: Option<usize>,
    pub file_config: Option<GenerationFileConfig>,
}

impl GenerationOverrides {
    pub fn with_temperature(mut self, temperature: Option<f32>) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_top_p(mut self, top_p: Option<f32>) -> Self {
        self.top_p = top_p;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: Option<usize>) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_repetition_penalty(mut self, penalty: Option<f32>) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    pub fn with_tool_choice(mut self, tool_choice: Option<ToolChoice>) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    pub fn with_reasoning_effort(mut self, reasoning_effort: Option<ReasoningEffort>) -> Self {
        self.reasoning_effort = reasoning_effort;
        self
    }

    pub fn with_stop_tokens(mut self, stop_tokens: Option<Vec<String>>) -> Self {
        self.stop_tokens = stop_tokens;
        self
    }

    pub fn with_logprobs(mut self, logprobs: Option<bool>) -> Self {
        self.logprobs = logprobs;
        self
    }

    pub fn with_top_logprobs(mut self, top_logprobs: Option<usize>) -> Self {
        self.top_logprobs = top_logprobs;
        self
    }

    pub fn with_seed(mut self, seed: Option<usize>) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_file_config(mut self, file_config: Option<GenerationFileConfig>) -> Self {
        self.file_config = file_config;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GenerationFileConfig;
    use crate::gguf::types::{GgufHeaders, GgufKVMeta, GgufTensorInfo, GgufType, GgufValue};
    use rstest::rstest;

    #[test]
    fn test_generation_config_defaults() {
        let default = GenerationConfig::default();
        assert_eq!(default.temperature, 0.8);
        assert_eq!(default.top_p, 0.9);
        assert_eq!(default.max_tokens, 32_000);
        assert_eq!(default.repetition_penalty, 1.15);
        assert_eq!(default.tool_choice, ToolChoice::Mode(ToolChoiceMode::Auto));
        assert_eq!(default.reasoning_effort, ReasoningEffort::High);
        assert_eq!(default.stop_tokens, None);
        assert_eq!(default.logprobs, false);
        assert_eq!(default.top_logprobs, None);
        assert_eq!(default.seed, 19);
    }

    #[rstest]
    #[case(GgufInfo {
        headers: GgufHeaders {
            path: "test_path".to_string(),
            magic: "GGUF".to_string(),
            version: 5342,
            tensor_count: 1,
            metadata_kv_count: 6,
        },
        kv_meta: vec![GgufKVMeta {
            key: "general.architecture".to_string(),
            value_type: GgufType::String,
            value: GgufValue::String("gpt2".to_string()),
        },
        GgufKVMeta {
            key: "sampling.temperature".to_string(),
            value_type: GgufType::Float32,
            value: GgufValue::Float32(0.87),
        },
        GgufKVMeta {
            key: "sampling.top_p".to_string(),
            value_type: GgufType::Float32,
            value: GgufValue::Float32(0.99),
        },
        GgufKVMeta {
            key: "general.max_tokens".to_string(),
            value_type: GgufType::Uint32,
            value: GgufValue::Uint32(64000),
        },
        GgufKVMeta {
            key: "sampling.repetition_penalty".to_string(),
            value_type: GgufType::Float32,
            value: GgufValue::Float32(1.158),
        },
        GgufKVMeta {
            key: "unknown.key".to_string(),
            value_type: GgufType::String,
            value: GgufValue::String("unknown.key".to_string()),
        }],
        tensor_info: vec![GgufTensorInfo {
            name: "test_tensor_name".to_string(),
            n_dims: 3,
            shapes: vec![1, 2, 3],
            ggml_type: 3,
            offset: 0,
        }],
        tensor_offset_start: 0,
    },
    GenerationConfig {
        temperature: 0.87,
        top_p: 0.99,
        max_tokens: 64_000,
        repetition_penalty: 1.158,
        tool_choice: ToolChoice::Mode(ToolChoiceMode::Auto),
        reasoning_effort: ReasoningEffort::High,
        stop_tokens: None,
        logprobs: false,
        top_logprobs: None,
        seed: 19,
    })]
    #[case(GgufInfo {
        headers: GgufHeaders {
            path: "test_path".to_string(),
            magic: "GGUF".to_string(),
            version: 5342,
            tensor_count: 1,
            metadata_kv_count: 6,
        },
        kv_meta: vec![GgufKVMeta {
            key: "general.architecture".to_string(),
            value_type: GgufType::String,
            value: GgufValue::String("gpt2".to_string()),
        },
        GgufKVMeta {
            key: "general.sampling.temp".to_string(),
            value_type: GgufType::Float32,
            value: GgufValue::Float32(0.87),
        },
        GgufKVMeta {
            key: "general.sampling.top_p".to_string(),
            value_type: GgufType::Float32,
            value: GgufValue::Float32(0.99),
        },
        GgufKVMeta {
            key: "general.max_tokens".to_string(),
            value_type: GgufType::Uint32,
            value: GgufValue::Uint32(64000),
        },
        GgufKVMeta {
            key: "general.sampling.repetition_penalty".to_string(),
            value_type: GgufType::Float32,
            value: GgufValue::Float32(1.158),
        },
        GgufKVMeta {
            key: "unknown.key".to_string(),
            value_type: GgufType::String,
            value: GgufValue::String("unknown.key".to_string()),
        }],
        tensor_info: vec![GgufTensorInfo {
            name: "test_tensor_name".to_string(),
            n_dims: 3,
            shapes: vec![1, 2, 3],
            ggml_type: 3,
            offset: 0,
        }],
        tensor_offset_start: 0,
    },
    GenerationConfig {
        temperature: 0.87,
        top_p: 0.99,
        max_tokens: 64_000,
        repetition_penalty: 1.158,
        tool_choice: ToolChoice::Mode(ToolChoiceMode::Auto),
        reasoning_effort: ReasoningEffort::High,
        stop_tokens: None,
        logprobs: false,
        top_logprobs: None,
        seed: 19,
    })]
    fn test_generation_config_from_gguf_metadata_success(
        #[case] gguf_info: GgufInfo,
        #[case] expected: GenerationConfig,
    ) {
        let gen_config = GenerationConfig::from_gguf_metadata(&gguf_info).unwrap();
        assert_eq!(gen_config, expected);
    }

    #[rstest]
    #[case(GgufInfo {
        headers: GgufHeaders {
            path: "test_path".to_string(),
            magic: "GGUF".to_string(),
            version: 5342,
            tensor_count: 1,
            metadata_kv_count: 5,
        },
        kv_meta: vec![GgufKVMeta {
            key: "sampling.temperature".to_string(),
            value_type: GgufType::Float32,
            value: GgufValue::Float32(0.87),
        },
        GgufKVMeta {
            key: "sampling.top_p".to_string(),
            value_type: GgufType::Float32,
            value: GgufValue::Float32(0.99),
        },
        GgufKVMeta {
            key: "general.max_tokens".to_string(),
            value_type: GgufType::Uint32,
            value: GgufValue::Uint32(64000),
        },
        GgufKVMeta {
            key: "sampling.repetition_penalty".to_string(),
            value_type: GgufType::Float32,
            value: GgufValue::Float32(1.158),
        },
        GgufKVMeta {
            key: "unknown.key".to_string(),
            value_type: GgufType::String,
            value: GgufValue::String("unknown.key".to_string()),
        }],
        tensor_info: vec![GgufTensorInfo {
            name: "test_tensor_name".to_string(),
            n_dims: 3,
            shapes: vec![1, 2, 3],
            ggml_type: 3,
            offset: 0,
        }],
        tensor_offset_start: 0,
    },
    "Could not find 'general.architecture' key in gguf metadata")]
    fn test_generation_config_from_gguf_metadata_failure(
        #[case] gguf_info: GgufInfo,
        #[case] expected_error_str: &str,
    ) {
        let error = GenerationConfig::from_gguf_metadata(&gguf_info).expect_err(
            "Expected GenerationConfig validation to fail, but it passed successfully.",
        );
        let error_message = error.to_string();
        assert!(
            error_message.contains(expected_error_str),
            "Expected error message to contain '{}', but got '{}'",
            expected_error_str,
            error_message
        );
    }

    #[test]
    fn test_generation_config_build_priority() {
        let builder = setup_test_builder();
        let gguf_info = setup_test_gguf_info();
        let config_default = builder.build();
        assert_eq!(config_default.temperature, 0.8);
        let builder = setup_test_builder();
        let file_config = setup_test_file_config();
        let config_file = builder.with_file_config(Some(file_config)).build();
        assert_eq!(config_file.temperature, 0.14);
        let builder = setup_test_builder();
        let config_gguf = builder.with_gguf_metadata(&gguf_info).unwrap().build();
        assert_eq!(config_gguf.temperature, 0.24);
        let builder = setup_test_builder();
        let file_config = setup_test_file_config();
        let config_with_file_and_gguf = builder
            .with_file_config(Some(file_config))
            .with_gguf_metadata(&gguf_info)
            .unwrap()
            .build();
        assert_eq!(config_with_file_and_gguf.temperature, 0.14);
        let builder = setup_test_builder();
        let overrides = setup_test_overrides();
        let config_overrides = builder.with_overrides(overrides).build();
        assert_eq!(config_overrides.temperature, 0.34);
        let builder = setup_test_builder();
        let file_config = setup_test_file_config();
        let overrides = setup_test_overrides();
        let config_file_and_overrides = builder
            .with_file_config(Some(file_config))
            .with_overrides(overrides)
            .build();
        assert_eq!(config_file_and_overrides.temperature, 0.34);
        let builder = setup_test_builder();
        let overrides = setup_test_overrides();
        let config_gguf_and_overrides = builder
            .with_gguf_metadata(&gguf_info)
            .unwrap()
            .with_overrides(overrides)
            .build();
        assert_eq!(config_gguf_and_overrides.temperature, 0.34);
        let builder = setup_test_builder();
        let file_config = setup_test_file_config();
        let overrides = setup_test_overrides();
        let config_file_gguf_and_overrides = builder
            .with_file_config(Some(file_config))
            .with_gguf_metadata(&gguf_info)
            .unwrap()
            .with_overrides(overrides)
            .build();
        assert_eq!(config_file_gguf_and_overrides.temperature, 0.34);
        let builder = setup_test_builder();
        let file_config = setup_test_file_config();
        let overrides = setup_test_overrides();
        let config_explicit = builder
            .with_file_config(Some(file_config))
            .with_overrides(overrides)
            .temperature(Some(0.44))
            .build();
        assert_eq!(config_explicit.temperature, 0.44);
    }

    fn setup_test_file_config() -> GenerationFileConfig {
        GenerationFileConfig {
            temperature: Some(0.14),
            top_p: Some(0.49),
            max_tokens: Some(64_000),
            repetition_penalty: Some(1.158),
            tool_choice: Some(ToolChoice::Mode(ToolChoiceMode::Auto)),
            reasoning_effort: Some(ReasoningEffort::High),
            stop_tokens: None,
            logprobs: None,
            top_logprobs: None,
            seed: Some(119),
        }
    }

    fn setup_test_overrides() -> GenerationOverrides {
        GenerationOverrides {
            temperature: Some(0.34),
            top_p: Some(0.99),
            max_tokens: Some(64_000),
            repetition_penalty: Some(1.158),
            tool_choice: Some(ToolChoice::Mode(ToolChoiceMode::Auto)),
            reasoning_effort: Some(ReasoningEffort::High),
            stop_tokens: None,
            logprobs: None,
            top_logprobs: None,
            seed: Some(119),
            file_config: None,
        }
    }

    fn setup_test_builder() -> GenerationConfigBuilder {
        GenerationConfigBuilder::default()
    }

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
                    key: "general.architecture".to_string(),
                    value_type: GgufType::String,
                    value: GgufValue::String("gpt2".to_string()),
                },
                GgufKVMeta {
                    key: "sampling.temperature".to_string(),
                    value_type: GgufType::Float32,
                    value: GgufValue::Float32(0.24),
                },
                GgufKVMeta {
                    key: "sampling.top_p".to_string(),
                    value_type: GgufType::Float32,
                    value: GgufValue::Float32(0.99),
                },
                GgufKVMeta {
                    key: "general.max_tokens".to_string(),
                    value_type: GgufType::Uint32,
                    value: GgufValue::Uint32(64000),
                },
                GgufKVMeta {
                    key: "sampling.repetition_penalty".to_string(),
                    value_type: GgufType::Float32,
                    value: GgufValue::Float32(1.158),
                },
                GgufKVMeta {
                    key: "unknown.key".to_string(),
                    value_type: GgufType::String,
                    value: GgufValue::String("unknown.key".to_string()),
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

    #[test]
    fn test_generation_overrides_chain_calls() {
        let overrides = GenerationOverrides::default();
        assert_eq!(overrides.top_p, None);
        let overrides_with_top_p = overrides.with_top_p(Some(0.49));
        assert_eq!(overrides_with_top_p.top_p, Some(0.49));
        let overrides_with_top_p_and_temperature =
            overrides_with_top_p.with_temperature(Some(0.99));
        assert_eq!(overrides_with_top_p_and_temperature.top_p, Some(0.49));
        assert_eq!(overrides_with_top_p_and_temperature.temperature, Some(0.99));
        let file_config = setup_test_file_config();

        let overrides_with_file_config =
            overrides_with_top_p_and_temperature.with_file_config(Some(file_config.clone()));
        assert_eq!(overrides_with_file_config.file_config, Some(file_config));
    }
}
