use candle_core::{DType, Tensor};
use candle_nn::kv_cache::ConcatKvCache;
use candle_transformers::quantized_var_builder::VarBuilder;

use crate::model::Lfm2Model;
use crate::model_config::{ModelArchitecture, ModelConfig};

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
}

impl ModelBuilder {
    pub fn new(config: ModelConfig, var_builder: VarBuilder, compute_dtype: DType) -> Self {
        Self {
            config,
            var_builder,
            compute_dtype
        }
    }

    pub fn build(self) -> anyhow::Result<Box<dyn Model>> {
        match self.config.architecture {
            ModelArchitecture::Lfm2 => {
                anyhow::Ok(Box::new(Lfm2Model::load(self.config, self.var_builder, self.compute_dtype)?))
            }
            ModelArchitecture::Llama => anyhow::bail!("Not implemented yet"),
        }
    }
}
