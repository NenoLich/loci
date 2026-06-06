use crate::gguf::GgufInfo;
use crate::api::types::{ToolChoice, ToolChoiceMode, ReasoningEffort};
use crate::config::GenerationFileConfig;

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

impl GenerationConfig {
    /// Create a new builder for generation config
    pub fn builder() -> GenerationConfigBuilder {
        GenerationConfigBuilder::default()
    }

    /// Resolve from GGUF metadata
    pub fn from_gguf_metadata(gguf_info: &GgufInfo) -> anyhow::Result<Self> {
        let mut temperature = GenerationConfigDefaults::TEMPERATURE;
        let mut max_tokens = GenerationConfigDefaults::MAX_TOKENS;
        let mut top_p = GenerationConfigDefaults::TOP_P;
        let mut repetition_penalty = GenerationConfigDefaults::REPETITION_PENALTY;
        let mut tool_choice = GenerationConfigDefaults::TOOL_CHOICE;
        let mut reasoning_effort = GenerationConfigDefaults::REASONING_EFFORT;
        let mut stop_tokens = GenerationConfigDefaults::STOP_TOKENS;
        let mut logprobs = GenerationConfigDefaults::LOGPROBS;
        let mut top_logprobs = GenerationConfigDefaults::TOP_LOGPROBS;
        let mut seed = GenerationConfigDefaults::SEED;

        let metadata = &gguf_info.kv_meta;

        let _architecture = metadata
            .iter()
            .find(|entry| entry.key == "general.architecture")
            .and_then(|entry| entry.value.as_string())
            .ok_or_else(|| anyhow::anyhow!("Could not find 'general.architecture' key in gguf metadata"))?;

        for entry in metadata {
            match entry.key.as_str() {
                "sampling.temperature" | "general.sampling.temp" | "general.sampling.temperature" => {
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
    pub fn temperature(mut self, temp: Option<f32>) -> Self {
        self.temperature = temp;
        self
    }

    /// Set explicit top_p (CLI argument)
    pub fn top_p(mut self, top_p: Option<f32>) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set explicit max_tokens (CLI argument)
    pub fn max_tokens(mut self, tokens: Option<usize>) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set explicit repetition_penalty (CLI argument)
    pub fn repetition_penalty(mut self, penalty: Option<f32>) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Set explicit tool_choice (CLI argument)
    pub fn tool_choice(mut self, tool_choice: Option<ToolChoice>) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    /// Set explicit reasoning_effort (CLI argument)
    pub fn reasoning_effort(mut self, effort: Option<ReasoningEffort>) -> Self {
        self.reasoning_effort = effort;
        self
    }

    /// Set explicit stop_tokens (CLI argument)
    pub fn stop_tokens(mut self, stop_tokens: Option<Vec<String>>) -> Self {
        self.stop_tokens = stop_tokens;
        self
    }

    /// Set explicit logprobs (CLI argument)
    pub fn logprobs(mut self, logprobs: Option<bool>) -> Self {
        self.logprobs = logprobs;
        self
    }

    /// Set explicit top_logprobs (CLI argument)
    pub fn top_logprobs(mut self, top_logprobs: Option<usize>) -> Self {
        self.top_logprobs = top_logprobs;
        self
    }

    /// Set explicit seed (CLI argument)
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

    pub fn with_file_config(mut self, config: Option<GenerationFileConfig>) -> Self {
        self.file_config = config;
        self
    }

    /// Resolve with priority: explicit > gguf > defaults
    pub fn build(self) -> GenerationConfig {
        GenerationConfig {
            temperature: self
                .temperature
                .or_else(|| self.file_config.as_ref().and_then(|c| c.temperature))
                .or_else(|| self.gguf_config.as_ref().map(|c| c.temperature))
                .unwrap_or(GenerationConfigDefaults::TEMPERATURE),
            top_p: self
                .top_p
                .or_else(|| self.file_config.as_ref().and_then(|c| c.top_p))
                .or_else(|| self.gguf_config.as_ref().map(|c| c.top_p))
                .unwrap_or(GenerationConfigDefaults::TOP_P),
            max_tokens: self
                .max_tokens
                .or_else(|| self.file_config.as_ref().and_then(|c| c.max_tokens))
                .or_else(|| self.gguf_config.as_ref().map(|c| c.max_tokens))
                .unwrap_or(GenerationConfigDefaults::MAX_TOKENS),
            repetition_penalty: self
                .repetition_penalty
                .or_else(|| self.file_config.as_ref().and_then(|c| c.repetition_penalty))
                .or_else(|| self.gguf_config.as_ref().map(|c| c.repetition_penalty))
                .unwrap_or(GenerationConfigDefaults::REPETITION_PENALTY),
            tool_choice: self
                .tool_choice
                .or_else(|| self.file_config.as_ref().and_then(|c| c.tool_choice.clone()))
                .or_else(|| self.gguf_config.as_ref().map(|c| c.tool_choice.clone()))
                .unwrap_or(GenerationConfigDefaults::TOOL_CHOICE),
            reasoning_effort: self
                .reasoning_effort
                .or_else(|| self.file_config.as_ref().and_then(|c| c.reasoning_effort.clone()))
                .or_else(|| self.gguf_config.as_ref().map(|c| c.reasoning_effort.clone()))
                .unwrap_or(GenerationConfigDefaults::REASONING_EFFORT),
            stop_tokens: self
                .stop_tokens
                .or_else(|| self.file_config.as_ref().and_then(|c| c.stop_tokens.clone()))
                .or_else(|| self.gguf_config.as_ref().and_then(|c| c.stop_tokens.clone())),
            logprobs: self
                .logprobs
                .or_else(|| self.file_config.as_ref().and_then(|c| c.logprobs))
                .or_else(|| self.gguf_config.as_ref().map(|c| c.logprobs))
                .unwrap_or(GenerationConfigDefaults::LOGPROBS),
            top_logprobs: self
                .top_logprobs
                .or_else(|| self.file_config.as_ref().and_then(|c| c.top_logprobs))
                .or_else(|| self.gguf_config.as_ref().and_then(|c| c.top_logprobs)),
            seed: self
                .seed
                .or_else(|| self.file_config.as_ref().and_then(|c| c.seed))
                .unwrap_or(GenerationConfigDefaults::SEED),
        }
    }
}

/// Default generation parameters
pub struct GenerationConfigDefaults;

impl GenerationConfigDefaults {
    pub const TEMPERATURE: f32 = 0.8;
    pub const MAX_TOKENS: usize = 32_000;
    pub const TOP_P: f32 = 0.9;
    pub const REPETITION_PENALTY: f32 = 1.15;
    pub const TOOL_CHOICE: ToolChoice = ToolChoice::Mode(ToolChoiceMode::Auto);
    pub const REASONING_EFFORT: ReasoningEffort = ReasoningEffort::High;
    pub const STOP_TOKENS: Option<Vec<String>> = None;
    pub const LOGPROBS: bool = false;
    pub const TOP_LOGPROBS: Option<usize> = None;
    pub const SEED: usize = 19;
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
    pub fn new(
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
        file_config: Option<GenerationFileConfig>,
    ) -> Self {
        Self {
            temperature,
            top_p,
            max_tokens,
            repetition_penalty,
            tool_choice,
            reasoning_effort,
            stop_tokens,
            logprobs,
            top_logprobs,
            seed,
            file_config,
        }
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = Some(penalty);
        self
    }

    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    pub fn reasoning_effort(mut self, reasoning_effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(reasoning_effort);
        self
    }

    pub fn stop_tokens(mut self, stop_tokens: Vec<String>) -> Self {
        self.stop_tokens = Some(stop_tokens);
        self
    }

    pub fn logprobs(mut self, logprobs: bool) -> Self {
        self.logprobs = Some(logprobs);
        self
    }

    pub fn top_logprobs(mut self, top_logprobs: usize) -> Self {
        self.top_logprobs = Some(top_logprobs);
        self
    }

    pub fn seed(mut self, seed: usize) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn file_config(mut self, file_config: GenerationFileConfig) -> Self {
        self.file_config = Some(file_config);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_explicit_over_defaults() {
        let config = GenerationConfig::builder()
            .temperature(Some(0.5))
            .max_tokens(Some(200))
            .build();

        assert_eq!(config.temperature, Some(0.5));
        assert_eq!(config.max_tokens, Some(200));
    }

    #[test]
    fn test_defaults_when_nothing_specified() {
        let config = GenerationConfig::builder().build();

        assert_eq!(config.temperature, Some(GenerationConfigDefaults::TEMPERATURE));
        assert_eq!(config.max_tokens, Some(GenerationConfigDefaults::MAX_TOKENS));
    }
}
