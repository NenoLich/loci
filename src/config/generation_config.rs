use crate::gguf::GgufInfo;

use std::rc::Rc;

/// Generation parameters with priority: CLI > GGUF metadata > defaults
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub seed: u64,
}

impl GenerationConfig {
    /// Create a new builder for generation config
    pub fn builder() -> GenerationConfigBuilder {
        GenerationConfigBuilder::default()
    }

    /// Resolve from GGUF metadata
    pub fn from_gguf_metadata(gguf_info: Rc<GgufInfo>) -> anyhow::Result<Self> {
        let mut temperature = GenerationConfigDefaults::TEMPERATURE;
        let mut max_tokens = GenerationConfigDefaults::MAX_TOKENS;
        let mut top_p = GenerationConfigDefaults::TOP_P;

        let metadata = &gguf_info.kv_meta;

        let _architecture = metadata
            .iter()
            .find(|entry| entry.key == "general.architecture")
            .and_then(|entry| entry.value.as_string())
            .ok_or_else(|| anyhow::anyhow!("Could not find 'general.architecture' key in gguf metadata"))?;

        for entry in metadata {
            match entry.key.as_str() {
                "sampling.temperature" | "general.sampling.temp" | "general.sampling.temperature" => {
                    if let Some(value) = entry.value.as_f64() {
                        temperature = value;
                    }
                }
                "sampling.top_p" | "general.sampling.top_p" => {
                    if let Some(value) = entry.value.as_f64() {
                        top_p = value;
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
            seed: GenerationConfigDefaults::SEED,
        })
    }
}

/// Builder for GenerationConfig with priority resolution
#[derive(Debug, Default)]
pub struct GenerationConfigBuilder {
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<usize>,
    seed: Option<u64>,
    gguf_config: Option<GenerationConfig>,
}

impl GenerationConfigBuilder {
    /// Set explicit temperature (CLI argument)
    pub fn temperature(mut self, temp: Option<f64>) -> Self {
        self.temperature = temp;
        self
    }

    /// Set explicit top_p (CLI argument)
    pub fn top_p(mut self, top_p: Option<f64>) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set explicit max_tokens (CLI argument)
    pub fn max_tokens(mut self, tokens: Option<usize>) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set explicit seed (CLI argument)
    pub fn seed(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        self
    }

    /// Load from GGUF metadata (fallback source)
    pub fn with_gguf_metadata(mut self, gguf_info: Rc<GgufInfo>) -> anyhow::Result<Self> {
        self.gguf_config = Some(GenerationConfig::from_gguf_metadata(gguf_info)?);
        Ok(self)
    }

    /// Resolve with priority: explicit > gguf > defaults
    pub fn build(self) -> GenerationConfig {
        GenerationConfig {
            temperature: self
                .temperature
                .or_else(|| self.gguf_config.as_ref().map(|c| c.temperature as f64))
                .unwrap_or(GenerationConfigDefaults::TEMPERATURE),
            top_p: self
                .top_p
                .or_else(|| self.gguf_config.as_ref().map(|c| c.top_p as f64))
                .unwrap_or(GenerationConfigDefaults::TOP_P),
            max_tokens: self
                .max_tokens
                .or_else(|| self.gguf_config.as_ref().map(|c| c.max_tokens))
                .unwrap_or(GenerationConfigDefaults::MAX_TOKENS),
            seed: self
                .seed
                .unwrap_or(GenerationConfigDefaults::SEED),
        }
    }
}

/// Default generation parameters
pub struct GenerationConfigDefaults;

impl GenerationConfigDefaults {
    pub const TEMPERATURE: f64 = 0.8;
    pub const MAX_TOKENS: usize = 32_000;
    pub const TOP_P: f64 = 0.9;
    pub const SEED: u64 = 19;
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
