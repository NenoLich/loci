use candle_core::{DType, Tensor};
use candle_nn::kv_cache::ConcatKvCache;
use candle_transformers::quantized_var_builder::VarBuilder;

use crate::config::{ModelArchitecture, ModelConfig};
use crate::model::{Lfm2Model, Deepseek2Model};

pub trait Model {
    fn forward(
        &self,
        input: &Tensor,
        cache: &mut Vec<Option<MixedCache>>,
        pos: usize,
        use_flash: bool,
    ) -> anyhow::Result<Tensor>;
    fn init_cache(&self) -> anyhow::Result<Vec<Option<MixedCache>>>;
}

pub enum MixedCache {
    KvCache(ConcatKvCache),
    ConvCache(Tensor),
}

pub struct ModelBuilder {
    pub config: ModelConfig,
    pub var_builder: VarBuilder,
    pub compute_dtype: DType,
    pub max_seq_len: usize,
    pub conv_on_cpu: bool,
}

impl ModelBuilder {
    pub fn new(config: ModelConfig, var_builder: VarBuilder, compute_dtype: DType, max_seq_len: usize, conv_on_cpu: bool) -> Self {
        Self {
            config,
            var_builder,
            compute_dtype,
            max_seq_len,
            conv_on_cpu,
        }
    }

    pub fn build(self) -> anyhow::Result<Box<dyn Model>> {
        match self.config.architecture {
            ModelArchitecture::Lfm2 => {
                anyhow::Ok(Box::new(Lfm2Model::load(self.config, self.var_builder, self.compute_dtype, self.max_seq_len, self.conv_on_cpu)?))
            }
            ModelArchitecture::Deepseek2 => {
                anyhow::Ok(Box::new(Deepseek2Model::load(self.config, self.var_builder, self.compute_dtype, self.max_seq_len)?))
            },
        }
    }
}