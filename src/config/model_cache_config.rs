use std::path::{PathBuf, Path};

use crate::config::CacheFileConfig;

pub struct ModelCacheConfigDefaults;

impl ModelCacheConfigDefaults {
    pub const CACHE_DIR: &'static str = "model_cache";
    pub const MAX_CACHE_SIZE: u64 = 16_000_000_000;
    pub const MIN_CACHE_TOKENS: usize = 512;
}

#[derive(Debug, Clone)]
pub struct ModelCacheConfig {
    pub cache_dir: PathBuf,
    pub max_cache_size: u64,
    pub min_cache_tokens: usize,
}

impl Default for ModelCacheConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl ModelCacheConfig {
    pub fn builder() -> ModelCacheConfigBuilder {
        ModelCacheConfigBuilder::default()
    }
}

#[derive(Default)]
pub struct ModelCacheConfigBuilder {
    cache_dir: Option<PathBuf>,
    max_cache_size: Option<u64>,
    min_cache_tokens: Option<usize>,
    file_config: Option<CacheFileConfig>
}

impl ModelCacheConfigBuilder {
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
    pub fn with_file_config(mut self, config: Option<CacheFileConfig>) -> Self {
        self.file_config = config;
        self
    }

    pub fn build(self) -> ModelCacheConfig {
        ModelCacheConfig {
            cache_dir: self.cache_dir
                .or_else(|| self.file_config.as_ref().and_then(|c| {
                    c.cache_dir.as_ref().and_then(|cache_dir_str| Some(PathBuf::from(cache_dir_str)))
                }))
                .unwrap_or_else(|| PathBuf::from(ModelCacheConfigDefaults::CACHE_DIR.to_string())),
            max_cache_size: self.max_cache_size
                .or_else(|| self.file_config.as_ref().and_then(|c| c.max_cache_size))
                .unwrap_or(ModelCacheConfigDefaults::MAX_CACHE_SIZE),
            min_cache_tokens: self.min_cache_tokens
                .or_else(|| self.file_config.as_ref().and_then(|c| c.min_cache_tokens))
                .unwrap_or(ModelCacheConfigDefaults::MIN_CACHE_TOKENS),
        }
    }
}