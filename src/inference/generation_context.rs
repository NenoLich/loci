use crate::config::ModelCacheConfig;
use crate::error::LociError;
use crate::inference::{
    CacheLoader, CacheMetadata, FileCacheLoader, ModelCacheManager, ModelCacheManagerInterface,
};
use crate::model::{MixedCache, ModelCacheInfo, ModelCacheType};
use crate::types::ModelCacheFragmentation;
use candle_core::Device;
use tracing::{debug, error, warn};

pub struct GenerationContext {
    pub model: String,
    pub token_ids: Vec<u32>,
    pub active_cache: Vec<Option<MixedCache>>,
    pub block_boundary_conv_cache: Vec<Vec<Option<MixedCache>>>,
    pub prefix_caching: bool,
    pub cache_type: ModelCacheType,
    pub cache_manager: Box<dyn ModelCacheManagerInterface + Send + Sync>,
    pub cache_loader: Box<dyn CacheLoader + Send + Sync>,
    pub cache_metadata: Vec<CacheMetadata>,
    pub model_cache_fragmentation: ModelCacheFragmentation,
    pub model_cache_seq_len_dim: usize,
    pub model_layers_count: usize,
}

impl GenerationContext {
    pub fn new(
        model: &str,
        model_cache_config: Option<ModelCacheConfig>,
        cache_info: ModelCacheInfo,
    ) -> Result<Self, LociError> {
        let model_cache_config = model_cache_config.unwrap_or_default();
        let prefix_caching = model_cache_config.prefix_caching;
        let model_cache_fragmentation = match model_cache_config.fragmentation {
            ModelCacheFragmentation::BlockWise { block_size } => {
                ModelCacheFragmentation::BlockWise {
                    block_size: block_size.max(cache_info.cache_block_size_hint),
                }
            }
            ModelCacheFragmentation::TokenWise => {
                if cache_info.cache_block_size_hint > 1 {
                    ModelCacheFragmentation::BlockWise {
                        block_size: cache_info.cache_block_size_hint,
                    }
                } else {
                    ModelCacheFragmentation::TokenWise
                }
            }
        };
        let cache_manager = ModelCacheManager::builder(model)
            .with_config(model_cache_config)
            .build();
        let cache_metadata = if prefix_caching {
            cache_manager.load_cache_metadata(&model_cache_fragmentation)?
        } else {
            Vec::new()
        };
        let cache_loader = Box::new(FileCacheLoader);

        Ok(Self {
            model: model.to_string(),
            token_ids: Vec::new(),
            active_cache: Vec::new(),
            block_boundary_conv_cache: Vec::new(),
            prefix_caching,
            cache_type: cache_info.cache_type,
            cache_manager: Box::new(cache_manager),
            cache_loader,
            cache_metadata,
            model_cache_fragmentation,
            model_cache_seq_len_dim: cache_info.cache_seq_len_dim,
            model_layers_count: cache_info.n_layers,
        })
    }

    pub fn update(&mut self, input_tokens: Vec<u32>) -> Result<(), LociError> {
        let input_tokens_len = input_tokens.len();
        if !self.active_cache.is_empty()
            && let ModelCacheType::MixedWithConv { conv_l_cache } = self.cache_type
        {
            self.update_conv_cache(input_tokens_len, conv_l_cache)?;
        }
        self.token_ids.extend(input_tokens);

        Ok(())
    }

    fn update_conv_cache(
        &mut self,
        input_tokens_len: usize,
        conv_l_cache: usize,
    ) -> Result<(), LociError> {
        if self.prefix_caching
            && let ModelCacheFragmentation::BlockWise { block_size } =
                self.model_cache_fragmentation
        {
            let added_token_id_len = self.token_ids.len();
            let total_token_id_len = added_token_id_len + input_tokens_len;
            let stored_block_count = self.block_boundary_conv_cache.len();
            let estimated_block_count = total_token_id_len.saturating_div(block_size);
            if stored_block_count < estimated_block_count {
                let new_block_count = estimated_block_count - stored_block_count;
                let block_boundary_input_indices = (stored_block_count..estimated_block_count)
                    .map(|count| {
                        (
                            count - stored_block_count,
                            ((count + 1) * block_size) - added_token_id_len,
                        )
                    })
                    .collect::<Vec<(usize, usize)>>();
                let mut new_block_boundary_conv_cache =
                    vec![vec![None; self.model_layers_count]; new_block_count];

                for (i, layer_cache) in self.active_cache.iter_mut().enumerate() {
                    if let Some(MixedCache::ConvCache(tensor)) = layer_cache {
                        let tensor_len = tensor
                            .dim(self.model_cache_seq_len_dim)
                            .map_err(|e| LociError::Cache(e.to_string()))?;
                        for (block_idx, block_boundary_idx) in &block_boundary_input_indices {
                            let boundary_start = tensor_len
                                .saturating_sub(input_tokens_len)
                                .saturating_add(*block_boundary_idx)
                                .saturating_sub(conv_l_cache);
                            let boundary_cache = tensor
                                .narrow(self.model_cache_seq_len_dim, boundary_start, conv_l_cache)
                                .map_err(|e| LociError::Cache(e.to_string()))?;
                            new_block_boundary_conv_cache[*block_idx][i] =
                                Some(MixedCache::ConvCache(boundary_cache));
                        }
                        let trim_start = tensor_len.saturating_sub(conv_l_cache);
                        *tensor = tensor
                            .narrow(self.model_cache_seq_len_dim, trim_start, conv_l_cache)
                            .map_err(|e| LociError::Cache(e.to_string()))?;
                    }
                }
                self.block_boundary_conv_cache
                    .extend(new_block_boundary_conv_cache);
            } else {
                self.update_active_conv_cache(conv_l_cache)?;
            }
        } else {
            self.update_active_conv_cache(conv_l_cache)?;
        }

        Ok(())
    }

    fn update_active_conv_cache(&mut self, conv_l_cache: usize) -> Result<(), LociError> {
        for layer_cache in self.active_cache.iter_mut() {
            if let Some(MixedCache::ConvCache(tensor)) = layer_cache {
                let tensor_len = tensor
                    .dim(self.model_cache_seq_len_dim)
                    .map_err(|e| LociError::Cache(e.to_string()))?;
                let trim_start = tensor_len.saturating_sub(conv_l_cache);
                *tensor = tensor
                    .narrow(self.model_cache_seq_len_dim, trim_start, conv_l_cache)
                    .map_err(|e| LociError::Cache(e.to_string()))?;
            }
        }
        Ok(())
    }

    pub fn match_cache(
        &mut self,
        prompt_token_ids: &[u32],
        min_prefill_tokens: usize,
        device: &Device,
        conv_on_cpu: bool,
    ) -> Result<usize, LociError> {
        if !self.prefix_caching {
            return Ok(0);
        }

        let prompt_token_len = prompt_token_ids.len();
        let max_cache_len = prompt_token_len.saturating_sub(min_prefill_tokens);
        let mut highest_cache_match_len = self
            .token_ids
            .iter()
            .zip(prompt_token_ids.iter())
            .take_while(|(a, b)| a == b)
            .count();
        // Only return fully matched active cache if cache is not empty, input is not subslice of cache and cache is fully matched
        if highest_cache_match_len != 0 {
            // Avoid when input is fully overlap with cache, we need to pass at least one token to model forward
            if highest_cache_match_len > max_cache_len {
                let cache_len = self.narrow_active_cache(max_cache_len)?;
                debug!("Matched cache length: {}", cache_len);

                return Ok(cache_len);
            }
            if highest_cache_match_len == self.token_ids.len() {
                debug!("Matched cache length: {}", highest_cache_match_len);

                return Ok(highest_cache_match_len);
            }
            if max_cache_len < self.cache_manager.min_cache_tokens() {
                let cache_len = self.narrow_active_cache(highest_cache_match_len)?;
                debug!("Matched cache length: {}", cache_len);

                return Ok(cache_len);
            }
        }

        let mut active_cache_source = true;
        let mut cache_file_path_with_highest_match = None;
        for cache_metadata in &self.cache_metadata {
            if cache_metadata.model != self.model {
                continue;
            }
            let cache_match_len = cache_metadata
                .token_ids
                .iter()
                .zip(prompt_token_ids.iter())
                .take_while(|(a, b)| a == b)
                .count();
            if cache_match_len > highest_cache_match_len
                && cache_match_len >= self.cache_manager.min_cache_tokens()
            {
                highest_cache_match_len = cache_match_len;
                active_cache_source = false;
                cache_file_path_with_highest_match = Some(cache_metadata.cache_file_path.clone());
            }
        }
        // Avoid when input is fully overlap with cache, we need to pass at least one token to model forward
        highest_cache_match_len = highest_cache_match_len.min(max_cache_len);

        let result = match (
            highest_cache_match_len,
            active_cache_source,
            cache_file_path_with_highest_match,
        ) {
            (0, _, _) => 0,
            (_, true, _) => {
                self.save_active_cache();
                let cache_len = self.narrow_active_cache(highest_cache_match_len)?;
                debug!("Matched cache length: {}", cache_len);

                cache_len
            }
            (_, false, Some(cache_file_path)) => {
                match self
                    .cache_loader
                    .load_mixed_cache(
                        cache_file_path,
                        highest_cache_match_len,
                        &self.model_cache_fragmentation,
                        self.model_cache_seq_len_dim,
                        device,
                        conv_on_cpu,
                    )
                    .map_err(|e| LociError::CacheLoad(format!("failed to load cache: {}", e)))
                {
                    Ok(loaded_mixed_cache) => {
                        debug!("Cache loaded from disk");
                        debug!(
                            "Matched cache length: {}",
                            loaded_mixed_cache.cached_token_length
                        );
                        self.save_active_cache();
                        self.token_ids.clear();
                        self.token_ids.extend_from_slice(
                            &prompt_token_ids[..loaded_mixed_cache.cached_token_length],
                        );
                        self.active_cache = loaded_mixed_cache.mixed_cache;
                        self.block_boundary_conv_cache =
                            loaded_mixed_cache.block_boundary_conv_cache;

                        loaded_mixed_cache.cached_token_length
                    }
                    Err(e) => {
                        error!("{}", e);
                        0
                    }
                }
            }
            (_, false, None) => 0,
        };

        Ok(result)
    }

    fn narrow_active_cache(&mut self, matched_cache_length: usize) -> Result<usize, LociError> {
        let (matched_cache_length, block_idx) = match &self.model_cache_fragmentation {
            ModelCacheFragmentation::BlockWise { block_size } => {
                let block_idx = matched_cache_length / block_size;
                (block_idx * block_size, block_idx.checked_sub(1))
            }
            ModelCacheFragmentation::TokenWise => (matched_cache_length, None),
        };

        for (layer_idx, cache) in self.active_cache.iter_mut().enumerate() {
            if let Some(MixedCache::KvCache(concat_kv_cache)) = cache {
                if concat_kv_cache.k().is_some() && concat_kv_cache.v().is_some() {
                    let k = concat_kv_cache.k_mut().unwrap();
                    *k = k
                        .narrow(self.model_cache_seq_len_dim, 0, matched_cache_length)
                        .map_err(|e| LociError::CacheLoad(e.to_string()))?;
                    let v = concat_kv_cache.v_mut().unwrap();
                    *v = v
                        .narrow(self.model_cache_seq_len_dim, 0, matched_cache_length)
                        .map_err(|e| LociError::CacheLoad(e.to_string()))?;
                }
                continue;
            }
            if let Some(MixedCache::ConvCache(conv_cache)) = cache {
                if let Some(idx) = block_idx {
                    let block_boundary_cache = if let Some(MixedCache::ConvCache(tensor)) =
                        &self.block_boundary_conv_cache[idx][layer_idx]
                    {
                        Ok(tensor.clone())
                    } else {
                        Err(LociError::Cache(
                            "block boundary cache not found".to_string(),
                        ))
                    }?;
                    *conv_cache = block_boundary_cache;
                } else {
                    *cache = None;
                }
            }
        }
        if let Some(idx) = block_idx {
            self.block_boundary_conv_cache.truncate(idx + 1);
        }
        self.token_ids.truncate(matched_cache_length);

        Ok(matched_cache_length)
    }

    pub fn reset_active_cache(
        &mut self,
        new_cache: Vec<Option<MixedCache>>,
        with_save: bool,
    ) -> Result<(), LociError> {
        if with_save {
            self.save_active_cache();
        }
        self.token_ids.clear();
        self.active_cache = new_cache;
        self.block_boundary_conv_cache.clear();

        Ok(())
    }

    pub fn save_active_cache(&mut self) {
        if !self.prefix_caching || self.token_ids.len() < self.cache_manager.min_cache_tokens() {
            return;
        }
        let block_size = match self.model_cache_fragmentation {
            ModelCacheFragmentation::BlockWise { block_size } => block_size,
            ModelCacheFragmentation::TokenWise => 1,
        };
        if self.token_ids.len() / block_size * block_size < self.cache_manager.min_cache_tokens() {
            return;
        }

        let error = match self.cache_manager.save_cache(
            &self.token_ids,
            &self.active_cache,
            self.model_cache_seq_len_dim,
            &self.model_cache_fragmentation,
            &self.block_boundary_conv_cache,
        ) {
            Ok(cache_file_path) => {
                self.cache_metadata.push(CacheMetadata {
                    model: self.model.clone(),
                    token_ids: self.token_ids.clone(),
                    cache_file_path,
                });
                self.cache_manager
                    .enforce_limits(&mut self.cache_metadata)
                    .err()
            }
            Err(e) => Some(e),
        };

        if let Some(e) = error {
            warn!("Failed to flush cache: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::model_cache::{
        LoadedMixedCache, MockCacheLoader, MockModelCacheManagerInterface,
    };
    use candle_core::Tensor;
    use candle_nn::kv_cache::ConcatKvCache;
    use rstest::rstest;
    use std::path::PathBuf;

    fn setup_test_generation_context(
        cache_manager: Box<dyn ModelCacheManagerInterface + Send + Sync>,
        cache_loader: Box<dyn CacheLoader + Send + Sync>,
        cache_meta: Vec<CacheMetadata>,
    ) -> GenerationContext {
        let active_cache = (0..2)
            .map(|_| {
                let mut layer_cache = ConcatKvCache::new(2);
                let k_tensor =
                    Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[1, 1, 5], &Device::Cpu)
                        .expect("k_tensor creation failed");
                let v_tensor =
                    Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[1, 1, 5], &Device::Cpu)
                        .expect("v_tensor creation failed");
                layer_cache
                    .append(&k_tensor, &v_tensor)
                    .expect("cache append failed");
                Some(MixedCache::KvCache(layer_cache))
            })
            .collect::<Vec<Option<MixedCache>>>();
        GenerationContext {
            model: "test_model".to_string(),
            token_ids: vec![1, 2, 3, 4, 5],
            active_cache,
            block_boundary_conv_cache: vec![],
            model_cache_seq_len_dim: 2,
            model_layers_count: 2,
            model_cache_fragmentation: ModelCacheFragmentation::TokenWise,
            prefix_caching: true,
            cache_type: ModelCacheType::FullAttn,
            cache_manager,
            cache_loader,
            cache_metadata: cache_meta,
        }
    }

    fn setup_expected_kv_cache(n_layers: usize, token_len: usize) -> Vec<Option<MixedCache>> {
        let mut cache = vec![];
        for _ in 0..n_layers {
            let mut layer_cache = ConcatKvCache::new(2);
            let k_tensor = Tensor::arange(0f64, token_len as f64, &Device::Cpu)
                .unwrap()
                .unsqueeze(0)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let v_tensor = Tensor::arange(0f64, token_len as f64, &Device::Cpu)
                .unwrap()
                .unsqueeze(0)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            layer_cache
                .append(&k_tensor, &v_tensor)
                .expect("cache append failed");
            cache.push(Some(MixedCache::KvCache(layer_cache)));
        }

        cache
    }

    #[rstest]
    #[case(
        vec![1, 2, 3, 4, 5],
        1,
        Box::new(MockModelCacheManagerInterface::new()),
        Box::new(MockCacheLoader::new()),
        vec![],
        setup_expected_kv_cache(2, 4),
        vec![1, 2, 3, 4]
    )]
    #[case(
        vec![1, 2, 3, 4, 5, 6],
        1,
        Box::new(MockModelCacheManagerInterface::new()),
        Box::new(MockCacheLoader::new()),
        vec![],
        setup_expected_kv_cache(2, 5),
        vec![1, 2, 3, 4, 5]
    )]
    #[case(
        vec![1, 2, 3, 4, 6],
        1,
        {
            let mut mock = MockModelCacheManagerInterface::new();
            mock.expect_min_cache_tokens().times(1).returning(|| 5);
            Box::new(mock)
        },
        Box::new(MockCacheLoader::new()),
        vec![],
        setup_expected_kv_cache(2, 4),
        vec![1, 2, 3, 4]
    )]
    #[case(
        vec![1, 2, 3, 4, 6],
        1,
        {
            let mut mock = MockModelCacheManagerInterface::new();
            mock.expect_min_cache_tokens().times(1).returning(|| 5);
            Box::new(mock)
        },
        Box::new(MockCacheLoader::new()),
        vec![],
        setup_expected_kv_cache(2, 4),
        vec![1, 2, 3, 4]
    )]
    #[case(
        vec![1, 2, 3, 4, 6],
        1,
        {
            let mut mock = MockModelCacheManagerInterface::new();
            mock.expect_min_cache_tokens().times(3).returning(|| 1);
            mock.expect_save_cache().times(1).returning(|_, _, _, _, _| Ok(PathBuf::from("test")));
            mock.expect_enforce_limits().times(1).returning(|_| Ok(()));
            Box::new(mock)
        },
        Box::new(MockCacheLoader::new()),
        vec![],
        setup_expected_kv_cache(2, 4),
        vec![1, 2, 3, 4]
    )]
    #[case(
        vec![1, 2, 3, 4, 6],
        1,
        {
            let mut mock = MockModelCacheManagerInterface::new();
            mock.expect_min_cache_tokens().times(3).returning(|| 1);
            mock.expect_save_cache().times(1).returning(|_, _, _, _, _| Ok(PathBuf::from("test")));
            mock.expect_enforce_limits().times(1).returning(|_| Ok(()));
            Box::new(mock)
        },
        Box::new(MockCacheLoader::new()),
        vec![CacheMetadata {
            model: "not_test_model".to_string(),
            token_ids: vec![1, 2, 3, 4, 5],
            cache_file_path: PathBuf::from("test"),
        }],
        setup_expected_kv_cache(2, 4),
        vec![1, 2, 3, 4]
    )]
    #[case(
        vec![1, 2, 3, 4, 6, 7],
        1,
        {
            let mut mock = MockModelCacheManagerInterface::new();
            mock.expect_min_cache_tokens().times(4).returning(|| 1);
            mock.expect_save_cache().times(1).returning(|_, _, _, _, _| Ok(PathBuf::from("test")));
            mock.expect_enforce_limits().times(1).returning(|_| Ok(()));
            Box::new(mock)
        },
        {
            let mut mock = MockCacheLoader::new();
            mock.expect_load_mixed_cache().times(1).returning(|_, _, _, _, _, _| Ok(LoadedMixedCache {
                mixed_cache: setup_expected_kv_cache(2, 4),
                block_boundary_conv_cache: vec![],
                cached_token_length: 4
            }));
            Box::new(mock)
        },
        vec![CacheMetadata {
            model: "test_model".to_string(),
            token_ids: vec![1, 2, 3, 4, 6],
            cache_file_path: PathBuf::from("test"),
        }],
        setup_expected_kv_cache(2, 4),
        vec![1, 2, 3, 4]
    )]
    fn test_match_cache_with_prefix_caching(
        #[case] prompt_token_ids: Vec<u32>,
        #[case] min_prefill_tokens: usize,
        #[case] cache_manager: Box<dyn ModelCacheManagerInterface + Send + Sync>,
        #[case] cache_loader: Box<dyn CacheLoader + Send + Sync>,
        #[case] cache_meta: Vec<CacheMetadata>,
        #[case] expected_active_cache: Vec<Option<MixedCache>>,
        #[case] expected_token_ids: Vec<u32>,
    ) {
        let mut gen_ctx = setup_test_generation_context(cache_manager, cache_loader, cache_meta);
        let matched_cache_len = gen_ctx
            .match_cache(&prompt_token_ids, min_prefill_tokens, &Device::Cpu, false)
            .expect("match_cache expected to succeed, but failed");
        let expected_cache_len = expected_token_ids.len();
        assert_eq!(matched_cache_len, expected_cache_len);
        assert_eq!(gen_ctx.token_ids.len(), expected_cache_len);
        let block_idx = match gen_ctx.model_cache_fragmentation {
            ModelCacheFragmentation::BlockWise { block_size } => {
                Some(expected_cache_len / block_size)
            }
            ModelCacheFragmentation::TokenWise => None,
        };
        if let Some(idx) = block_idx {
            assert_eq!(gen_ctx.block_boundary_conv_cache.len(), idx);
        }
        for (layer_idx, cache) in gen_ctx.active_cache.iter().enumerate() {
            if let Some(MixedCache::KvCache(concat_kv_cache)) = cache {
                let k = concat_kv_cache.k().expect("expected k to be Some");
                let v = concat_kv_cache.v().expect("expected v to be Some");
                assert!(matches!(
                    expected_active_cache[layer_idx].clone().unwrap(),
                    MixedCache::KvCache(..)
                ));
                if let MixedCache::KvCache(expected_cache) =
                    expected_active_cache[layer_idx].clone().unwrap()
                {
                    assert_eq!(
                        k.flatten_all().unwrap().to_vec1::<f64>().unwrap(),
                        expected_cache
                            .k()
                            .unwrap()
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<f64>()
                            .unwrap(),
                    );
                    assert_eq!(
                        v.flatten_all().unwrap().to_vec1::<f64>().unwrap(),
                        expected_cache
                            .v()
                            .unwrap()
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<f64>()
                            .unwrap(),
                    );
                }
            }
            if let Some(MixedCache::ConvCache(conv_cache)) = cache {
                if let Some(idx) = block_idx {
                    let block_boundary_cache = &gen_ctx.block_boundary_conv_cache[idx][layer_idx];
                    if let Some(MixedCache::ConvCache(tensor)) = block_boundary_cache {
                        assert_eq!(
                            tensor.dims()[gen_ctx.model_cache_seq_len_dim],
                            conv_cache.dims()[gen_ctx.model_cache_seq_len_dim]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_match_cache_prefix_caching_disabled() {
        let mut gen_ctx = setup_test_generation_context(
            Box::new(MockModelCacheManagerInterface::new()),
            Box::new(MockCacheLoader::new()),
            vec![],
        );
        gen_ctx.prefix_caching = false;
        let result = gen_ctx
            .match_cache(&[1, 2, 3, 4, 5], 1, &Device::Cpu, false)
            .unwrap();
        assert_eq!(result, 0);
    }

    #[rstest]
    #[case(
        vec![9, 10, 11, 12, 13],
        1,
        vec![],
        0usize,
        vec![1, 2, 3, 4, 5],
    )]
    fn test_match_cache_no_match_in_active(
        #[case] prompt: Vec<u32>,
        #[case] min_prefill: usize,
        #[case] cache_meta: Vec<CacheMetadata>,
        #[case] expected_result: usize,
        #[case] expected_token_ids: Vec<u32>,
    ) {
        let mut gen_ctx = setup_test_generation_context(
            Box::new(MockModelCacheManagerInterface::new()),
            Box::new(MockCacheLoader::new()),
            cache_meta,
        );
        gen_ctx.token_ids = vec![1, 2, 3, 4, 5];
        let result = gen_ctx
            .match_cache(&prompt, min_prefill, &Device::Cpu, false)
            .unwrap();
        assert_eq!(result, expected_result);
        assert_eq!(gen_ctx.token_ids, expected_token_ids);
    }

    #[test]
    fn test_match_cache_disk_load_from_metadata_without_active_match() {
        let mut mgr = MockModelCacheManagerInterface::new();
        mgr.expect_min_cache_tokens().times(..).returning(|| 1);
        mgr.expect_save_cache()
            .times(1)
            .returning(|_, _, _, _, _| Ok(PathBuf::from("test")));
        mgr.expect_enforce_limits().times(1).returning(|_| Ok(()));

        let mut loader = MockCacheLoader::new();
        loader
            .expect_load_mixed_cache()
            .times(1)
            .returning(|_, _, _, _, _, _| {
                Ok(LoadedMixedCache {
                    mixed_cache: setup_expected_kv_cache(2, 3),
                    block_boundary_conv_cache: vec![],
                    cached_token_length: 3,
                })
            });

        let mut gen_ctx = setup_test_generation_context(
            Box::new(mgr),
            Box::new(loader),
            vec![CacheMetadata {
                model: "test_model".to_string(),
                token_ids: vec![1, 2, 3, 9, 10],
                cache_file_path: PathBuf::from("test"),
            }],
        );
        gen_ctx.token_ids = vec![9, 10, 11, 12, 13];
        let result = gen_ctx
            .match_cache(&[1, 2, 3, 4, 5], 1, &Device::Cpu, false)
            .unwrap();
        assert_eq!(result, 3);
        assert_eq!(gen_ctx.token_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_match_cache_disk_load_failure() {
        let mut mgr = MockModelCacheManagerInterface::new();
        mgr.expect_min_cache_tokens().times(..).returning(|| 1);

        let mut loader = MockCacheLoader::new();
        loader
            .expect_load_mixed_cache()
            .times(1)
            .returning(|_, _, _, _, _, _| Err(anyhow::anyhow!("load failed")));

        let mut gen_ctx = setup_test_generation_context(
            Box::new(mgr),
            Box::new(loader),
            vec![CacheMetadata {
                model: "test_model".to_string(),
                token_ids: vec![1, 2, 3, 4, 5],
                cache_file_path: PathBuf::from("test"),
            }],
        );
        gen_ctx.token_ids = vec![9, 10, 11, 12, 13];
        let result = gen_ctx
            .match_cache(&[1, 2, 3, 4, 5], 1, &Device::Cpu, false)
            .unwrap();
        assert_eq!(result, 0);
    }

    #[test]
    fn test_match_cache_metadata_filtered_by_min_cache_tokens() {
        let mut mgr = MockModelCacheManagerInterface::new();
        mgr.expect_min_cache_tokens().times(..).returning(|| 5);

        let mut gen_ctx = setup_test_generation_context(
            Box::new(mgr),
            Box::new(MockCacheLoader::new()),
            vec![CacheMetadata {
                model: "test_model".to_string(),
                token_ids: vec![1, 2, 3],
                cache_file_path: PathBuf::from("test"),
            }],
        );
        gen_ctx.token_ids = vec![9, 10, 11, 12, 13];
        let result = gen_ctx
            .match_cache(&[1, 2, 3, 4, 5], 1, &Device::Cpu, false)
            .unwrap();
        assert_eq!(result, 0);
    }

    #[test]
    fn test_save_active_cache_below_min_cache_tokens() {
        let mut mgr = MockModelCacheManagerInterface::new();
        mgr.expect_min_cache_tokens().returning(|| 10);
        mgr.expect_save_cache().times(0);

        let mut gen_ctx =
            setup_test_generation_context(Box::new(mgr), Box::new(MockCacheLoader::new()), vec![]);
        gen_ctx.token_ids = vec![1, 2, 3, 4, 5];
        gen_ctx.save_active_cache();
    }

    #[test]
    fn test_narrow_active_cache_blockwise_with_conv() {
        let block_size = 4;
        let n_layers = 2;
        let seq_len_dim = 2;

        let mut block_boundary = vec![vec![None; n_layers]];
        for layer in 0..n_layers {
            let tensor = Tensor::from_slice(&[4.0f64, 5.0, 6.0], &[1, 1, 3], &Device::Cpu).unwrap();
            block_boundary[0][layer] = Some(MixedCache::ConvCache(tensor));
        }

        let mut active_cache = Vec::new();
        for _ in 0..n_layers {
            let tensor = Tensor::arange(0f64, 8f64, &Device::Cpu)
                .unwrap()
                .reshape(&[1, 1, 8])
                .unwrap();
            active_cache.push(Some(MixedCache::ConvCache(tensor)));
        }

        let mut gen_ctx = GenerationContext {
            model: "test_model".to_string(),
            token_ids: vec![1, 2, 3, 4, 5, 6, 7, 8],
            active_cache,
            block_boundary_conv_cache: block_boundary,
            model_cache_seq_len_dim: seq_len_dim,
            model_layers_count: n_layers,
            model_cache_fragmentation: ModelCacheFragmentation::BlockWise { block_size },
            prefix_caching: true,
            cache_type: ModelCacheType::FullAttn,
            cache_manager: Box::new(MockModelCacheManagerInterface::new()),
            cache_loader: Box::new(MockCacheLoader::new()),
            cache_metadata: vec![],
        };

        let result = gen_ctx.narrow_active_cache(6).unwrap();
        assert_eq!(result, 4);
        assert_eq!(gen_ctx.token_ids, vec![1, 2, 3, 4]);
        assert_eq!(gen_ctx.block_boundary_conv_cache.len(), 1);

        for cache in &gen_ctx.active_cache {
            if let Some(MixedCache::ConvCache(tensor)) = cache {
                assert_eq!(tensor.dims()[seq_len_dim], 3);
            }
        }
    }

    #[test]
    fn test_update_conv_cache_trim() {
        let conv_l_cache = 3;
        let seq_len_dim = 2;
        let n_layers = 2;

        let mut active_cache = Vec::new();
        for _ in 0..n_layers {
            let tensor = Tensor::arange(0f64, 10f64, &Device::Cpu)
                .unwrap()
                .reshape(&[1, 1, 10])
                .unwrap();
            active_cache.push(Some(MixedCache::ConvCache(tensor)));
        }

        let mut gen_ctx = GenerationContext {
            model: "test_model".to_string(),
            token_ids: vec![],
            active_cache,
            block_boundary_conv_cache: vec![],
            model_cache_seq_len_dim: seq_len_dim,
            model_layers_count: n_layers,
            model_cache_fragmentation: ModelCacheFragmentation::TokenWise,
            prefix_caching: false,
            cache_type: ModelCacheType::MixedWithConv { conv_l_cache },
            cache_manager: Box::new(MockModelCacheManagerInterface::new()),
            cache_loader: Box::new(MockCacheLoader::new()),
            cache_metadata: vec![],
        };

        gen_ctx.update(vec![1, 2, 3, 4, 5, 6, 7]).unwrap();
        assert_eq!(gen_ctx.token_ids, vec![1, 2, 3, 4, 5, 6, 7]);

        for cache in &gen_ctx.active_cache {
            if let Some(MixedCache::ConvCache(tensor)) = cache {
                assert_eq!(tensor.dims()[seq_len_dim], conv_l_cache);
            }
        }
    }

    #[test]
    fn test_update_conv_cache_block_boundary_extraction() {
        let block_size = 4;
        let conv_l_cache = 3;
        let seq_len_dim = 2;
        let n_layers = 2;

        let mut active_cache = Vec::new();
        for _ in 0..n_layers {
            let tensor = Tensor::arange(0f64, 14f64, &Device::Cpu)
                .unwrap()
                .reshape(&[1, 1, 14])
                .unwrap();
            active_cache.push(Some(MixedCache::ConvCache(tensor)));
        }

        let mut gen_ctx = GenerationContext {
            model: "test_model".to_string(),
            token_ids: vec![],
            active_cache,
            block_boundary_conv_cache: vec![],
            model_cache_seq_len_dim: seq_len_dim,
            model_layers_count: n_layers,
            model_cache_fragmentation: ModelCacheFragmentation::BlockWise { block_size },
            prefix_caching: true,
            cache_type: ModelCacheType::MixedWithConv { conv_l_cache },
            cache_manager: Box::new(MockModelCacheManagerInterface::new()),
            cache_loader: Box::new(MockCacheLoader::new()),
            cache_metadata: vec![],
        };

        gen_ctx
            .update(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            .unwrap();
        assert_eq!(gen_ctx.token_ids, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);

        let expected_blocks = 11usize / block_size;
        assert_eq!(gen_ctx.block_boundary_conv_cache.len(), expected_blocks);

        for cache in &gen_ctx.active_cache {
            if let Some(MixedCache::ConvCache(tensor)) = cache {
                assert_eq!(tensor.dims()[seq_len_dim], conv_l_cache);
            }
        }
    }
}
