use std::path::{Path, PathBuf};

use crate::config::CacheFileConfig;
use crate::types::ModelCacheFragmentation;

#[derive(Debug, Clone, PartialEq)]
pub struct ModelCacheConfig {
    pub prefix_caching: bool,
    pub cache_dir: PathBuf,
    pub max_cache_size: u64,
    pub min_cache_tokens: usize,
    pub fragmentation: ModelCacheFragmentation,
}

impl Default for ModelCacheConfig {
    fn default() -> Self {
        Self {
            prefix_caching: false,
            cache_dir: PathBuf::from("model_cache"),
            max_cache_size: 16_000_000_000,
            min_cache_tokens: 512,
            fragmentation: ModelCacheFragmentation::BlockWise { block_size: 32 },
        }
    }
}

impl ModelCacheConfig {
    pub fn builder() -> ModelCacheConfigBuilder {
        ModelCacheConfigBuilder::default()
    }
}

#[derive(Default)]
pub struct ModelCacheConfigBuilder {
    prefix_caching: Option<bool>,
    cache_dir: Option<PathBuf>,
    max_cache_size: Option<u64>,
    min_cache_tokens: Option<usize>,
    cache_block_size: Option<usize>,
    file_config: Option<CacheFileConfig>,
}

impl ModelCacheConfigBuilder {
    pub fn prefix_caching(mut self, prefix_caching: Option<bool>) -> Self {
        self.prefix_caching = prefix_caching;
        self
    }
    pub fn cache_dir(mut self, cache_dir: Option<impl AsRef<Path>>) -> Self {
        self.cache_dir = cache_dir.map(|path| PathBuf::from(path.as_ref()));
        self
    }

    pub fn max_cache_size(mut self, max_cache_size: Option<u64>) -> Self {
        self.max_cache_size = max_cache_size;
        self
    }

    pub fn min_cache_tokens(mut self, min_cache_tokens: Option<usize>) -> Self {
        self.min_cache_tokens = min_cache_tokens;
        self
    }
    pub fn cache_block_size(mut self, cache_block_size: Option<usize>) -> Self {
        self.cache_block_size = cache_block_size;
        self
    }
    pub fn with_file_config(mut self, config: Option<CacheFileConfig>) -> Self {
        self.file_config = config;
        self
    }

    pub fn build(self) -> ModelCacheConfig {
        let default = ModelCacheConfig::default();
        let fragmentation = if let Some(block_size) = self.cache_block_size {
            if block_size == 1 {
                ModelCacheFragmentation::TokenWise
            } else {
                ModelCacheFragmentation::BlockWise { block_size }
            }
        } else {
            self.file_config
                .as_ref()
                .and_then(|c| c.fragmentation.clone())
                .unwrap_or(default.fragmentation)
        };
        ModelCacheConfig {
            prefix_caching: self
                .prefix_caching
                .or_else(|| self.file_config.as_ref().and_then(|c| c.prefix_caching))
                .unwrap_or(default.prefix_caching),
            cache_dir: self
                .cache_dir
                .or_else(|| {
                    self.file_config
                        .as_ref()
                        .and_then(|c| c.cache_dir.as_ref().map(PathBuf::from))
                })
                .unwrap_or(default.cache_dir),
            max_cache_size: self
                .max_cache_size
                .or_else(|| self.file_config.as_ref().and_then(|c| c.max_cache_size))
                .unwrap_or(default.max_cache_size),
            min_cache_tokens: self
                .min_cache_tokens
                .or_else(|| self.file_config.as_ref().and_then(|c| c.min_cache_tokens))
                .unwrap_or(default.min_cache_tokens),
            fragmentation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn test_default() {
        let config = ModelCacheConfig::default();
        assert_eq!(config.prefix_caching, false);
        assert_eq!(config.cache_dir, PathBuf::from("model_cache"));
        assert_eq!(config.max_cache_size, 16_000_000_000);
        assert_eq!(config.min_cache_tokens, 512);
        assert_eq!(
            config.fragmentation,
            ModelCacheFragmentation::BlockWise { block_size: 32 }
        );
    }

    #[rstest]
    #[case(Some(5), ModelCacheFragmentation::BlockWise { block_size: 5 })]
    #[case(Some(1), ModelCacheFragmentation::TokenWise)]
    #[case(None, ModelCacheFragmentation::BlockWise { block_size: 32 })]
    fn test_fragmentation_resolution(
        #[case] cache_block_size: Option<usize>,
        #[case] expected: ModelCacheFragmentation,
    ) {
        let config = ModelCacheConfigBuilder::default()
            .cache_block_size(cache_block_size)
            .build();
        assert_eq!(config.fragmentation, expected);
    }

    #[test]
    fn test_build_priority() {
        let config = ModelCacheConfigBuilder::default().build();
        assert_eq!(
            config.fragmentation,
            ModelCacheFragmentation::BlockWise { block_size: 32 }
        );
        let config_with_file = ModelCacheConfigBuilder::default()
            .with_file_config(Some(CacheFileConfig {
                fragmentation: Some(ModelCacheFragmentation::TokenWise),
                prefix_caching: Some(true),
                cache_dir: Some("model_cache".to_string()),
                max_cache_size: Some(17_000_000_000),
                min_cache_tokens: Some(258),
            }))
            .build();
        assert_eq!(config_with_file.prefix_caching, true);
        assert_eq!(
            config_with_file.fragmentation,
            ModelCacheFragmentation::TokenWise
        );
        let config_with_file_and_explicit = ModelCacheConfigBuilder::default()
            .with_file_config(Some(CacheFileConfig {
                fragmentation: Some(ModelCacheFragmentation::TokenWise),
                prefix_caching: Some(true),
                cache_dir: Some("model_cache".to_string()),
                max_cache_size: Some(17_000_000_000),
                min_cache_tokens: Some(258),
            }))
            .min_cache_tokens(Some(64))
            .cache_dir(Some("explicit_cache_dir"))
            .prefix_caching(Some(false))
            .max_cache_size(Some(18_000_000_000))
            .build();
        assert_eq!(config_with_file_and_explicit.min_cache_tokens, 64);
        assert_eq!(
            config_with_file_and_explicit.cache_dir,
            PathBuf::from("explicit_cache_dir")
        );
        assert_eq!(config_with_file_and_explicit.prefix_caching, false);
        assert_eq!(config_with_file_and_explicit.max_cache_size, 18_000_000_000);
    }
}
