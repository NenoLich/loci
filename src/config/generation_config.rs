use crate::config::GenerationFileConfig;
use crate::gguf::GgufInfo;
use crate::types::{ReasoningEffort, ToolChoice, ToolChoiceMode};

/// Generation parameters with priority: CLI > GGUF metadata > defaults
#[derive(Debug, Clone)]
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
            .and_then(|entry| entry.value.as_string())
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

#[derive(Default)]
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
