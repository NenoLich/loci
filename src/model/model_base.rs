use candle_core::{DType, Tensor};
use candle_nn::kv_cache::ConcatKvCache;
use candle_transformers::quantized_var_builder::VarBuilder;
#[cfg(any(test, feature = "mock"))]
use mockall::automock;

use crate::config::{InferenceConfig, ModelArchitecture, ModelConfig};
use crate::error::LociError;
use crate::model::{Deepseek2Model, Lfm2Model};

#[derive(Debug, Clone)]
pub struct ModelCacheInfo {
    pub cache_type: ModelCacheType,
    pub cache_seq_len_dim: usize,
    pub cache_block_size_hint: usize,
    pub n_layers: usize,
}

#[cfg_attr(any(test, feature = "mock"), automock)]
pub trait Model {
    fn forward(
        &self,
        input: &Tensor,
        cache: &mut Vec<Option<MixedCache>>,
        pos: usize,
        use_flash: bool,
    ) -> anyhow::Result<Tensor>;
    fn init_cache(&self) -> anyhow::Result<Vec<Option<MixedCache>>>;
    fn cache_seq_len_dim(&self) -> usize;
    fn min_prefill_tokens(&self) -> usize {
        1
    }
    fn conv_on_cpu(&self) -> bool {
        false
    }
    fn cache_block_size_hint(&self) -> usize {
        1
    }
    fn model_cache_type(&self) -> ModelCacheType {
        ModelCacheType::FullAttn
    }

    fn n_layers(&self) -> usize;

    fn cache_info(&self) -> ModelCacheInfo {
        ModelCacheInfo {
            cache_type: self.model_cache_type(),
            cache_seq_len_dim: self.cache_seq_len_dim(),
            cache_block_size_hint: self.cache_block_size_hint(),
            n_layers: self.n_layers(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ModelCacheType {
    FullAttn,
    MixedWithConv { conv_l_cache: usize },
}

#[derive(Debug, Clone)]
pub enum MixedCache {
    KvCache(ConcatKvCache),
    ConvCache(Tensor),
}

impl MixedCache {
    pub fn as_conv_cache(&self) -> Option<&Tensor> {
        if let MixedCache::ConvCache(tensor) = self {
            Some(tensor)
        } else {
            None
        }
    }
}

pub struct ModelBuilder {
    pub config: ModelConfig,
    pub var_builder: VarBuilder,
    pub compute_dtype: DType,
    pub max_seq_len: usize,
    pub conv_on_cpu: bool,
}

impl ModelBuilder {
    pub fn new(
        config: ModelConfig,
        var_builder: VarBuilder,
        inference_config: &InferenceConfig,
    ) -> Self {
        Self {
            config,
            var_builder,
            compute_dtype: inference_config.dtype,
            max_seq_len: inference_config.max_seq_len,
            conv_on_cpu: inference_config.conv_on_cpu,
        }
    }

    pub fn build(self) -> Result<Box<dyn Model + Send + Sync>, LociError> {
        match self.config.architecture {
            ModelArchitecture::Lfm2 => Ok(Box::new(
                Lfm2Model::load(
                    self.config,
                    self.var_builder,
                    self.compute_dtype,
                    self.max_seq_len,
                    self.conv_on_cpu,
                )
                .map_err(|e| LociError::ModelLoad(e.to_string()))?,
            )),
            ModelArchitecture::Deepseek2 => Ok(Box::new(
                Deepseek2Model::load(
                    self.config,
                    self.var_builder,
                    self.compute_dtype,
                    self.max_seq_len,
                )
                .map_err(|e| LociError::ModelLoad(e.to_string()))?,
            )),
        }
    }
}
