use crate::config::ModelCacheConfig;
use crate::error::LociError;
use crate::model::MixedCache;
use crate::types::ModelCacheFragmentation;
use byteorder::{ByteOrder, LittleEndian};
use candle_core::{DType, Device, Tensor};
use candle_nn::kv_cache::ConcatKvCache;
#[cfg(test)]
use mockall::automock;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::path::PathBuf;
use tracing::{debug, error, info};
use uuid::Uuid;

struct CacheFileMetadata {
    cache_file_path: PathBuf,
    modified: std::time::SystemTime,
    file_size: u64,
}

pub struct CacheMetadata {
    pub model: String,
    pub token_ids: Vec<u32>,
    pub cache_file_path: PathBuf,
}

pub struct LoadedMixedCache {
    pub mixed_cache: Vec<Option<MixedCache>>,
    pub block_boundary_conv_cache: Vec<Vec<Option<MixedCache>>>,
    pub cached_token_length: usize,
}

pub struct ModelCacheManagerBuilder {
    config: Option<ModelCacheConfig>,
    model: String,
}

impl ModelCacheManagerBuilder {
    pub fn new(model: &str) -> Self {
        Self {
            config: None,
            model: model.to_string(),
        }
    }

    pub fn with_config(mut self, config: ModelCacheConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn build(self) -> ModelCacheManager {
        let config = self.config.unwrap_or_default();
        let cache_dir = config.cache_dir;
        let max_cache_size = config.max_cache_size;
        let min_cache_tokens = config.min_cache_tokens;

        ModelCacheManager {
            model: self.model,
            cache_dir,
            max_cache_size,
            min_cache_tokens,
        }
    }
}

#[cfg_attr(test, automock)]
pub trait ModelCacheManagerInterface {
    fn min_cache_tokens(&self) -> usize;
    fn save_cache(
        &self,
        token_ids: &[u32],
        cache: &[Option<MixedCache>],
        cache_seq_len_dim: usize,
        fragmentation: &ModelCacheFragmentation,
        block_boundary_conv_cache: &[Vec<Option<MixedCache>>],
    ) -> Result<PathBuf, LociError>;
    fn enforce_limits(&self, cache_meta: &mut Vec<CacheMetadata>) -> Result<(), LociError>;
}

#[derive(Default)]
pub struct ModelCacheManager {
    model: String,
    cache_dir: PathBuf,
    max_cache_size: u64,
    min_cache_tokens: usize,
}

impl ModelCacheManagerInterface for ModelCacheManager {
    fn min_cache_tokens(&self) -> usize {
        self.min_cache_tokens
    }
    fn save_cache(
        &self,
        token_ids: &[u32],
        cache: &[Option<MixedCache>],
        cache_seq_len_dim: usize,
        fragmentation: &ModelCacheFragmentation,
        block_boundary_conv_cache: &[Vec<Option<MixedCache>>],
    ) -> Result<PathBuf, LociError> {
        let (mut data, token_len_to_save, block_size) = match fragmentation {
            ModelCacheFragmentation::BlockWise { block_size } => (
                Vec::with_capacity(
                    cache.len() * 2 + 1 + (block_boundary_conv_cache.len() * cache.len()),
                ),
                token_ids.len() / (*block_size) * (*block_size),
                *block_size,
            ),
            ModelCacheFragmentation::TokenWise => {
                (Vec::with_capacity(cache.len() * 2 + 1), token_ids.len(), 1)
            }
        };
        let cache_file_path = self.cache_dir.join(format!(
            "cache-{}-blk-size-{}-{}.safetensors",
            self.model,
            block_size,
            Uuid::new_v4()
        ));

        let mut metadata = HashMap::new();
        metadata.insert("model_name".to_string(), self.model.clone());
        metadata.insert("n_layers".to_string(), cache.len().to_string());
        metadata.insert("fragmentation".to_string(), fragmentation.to_string());
        metadata.insert("block_size".to_string(), block_size.to_string());

        let token_ids_tensor = Tensor::from_slice(
            &token_ids[..token_len_to_save],
            token_len_to_save,
            &Device::Cpu,
        )
        .map_err(|e| LociError::Cache(format!("failed to create token_ids tensor: {}", e)))?;
        data.push(("token_ids".to_string(), token_ids_tensor));
        for (i, layer) in cache.iter().enumerate() {
            if let Some(MixedCache::KvCache(concat_kv_cache)) = layer.as_ref()
                && let (Some(k), Some(v)) = (concat_kv_cache.k(), concat_kv_cache.v())
            {
                let k_to_save = k
                    .narrow(cache_seq_len_dim, 0, token_len_to_save)
                    .map_err(|e| LociError::Cache(e.to_string()))?;
                let v_to_save = v
                    .narrow(cache_seq_len_dim, 0, token_len_to_save)
                    .map_err(|e| LociError::Cache(e.to_string()))?;
                data.push((format!("layer_{}_k", i), k_to_save));
                data.push((format!("layer_{}_v", i), v_to_save));
            }
        }
        if let ModelCacheFragmentation::BlockWise { .. } = fragmentation {
            for (block_idx, block_boundary_cache) in block_boundary_conv_cache.iter().enumerate() {
                for (layer_idx, layer_cache) in block_boundary_cache.iter().enumerate() {
                    if let Some(MixedCache::ConvCache(conv_cache_tensor)) = layer_cache.as_ref() {
                        data.push((
                            format!("block_{}_layer_{}_conv_cache", block_idx, layer_idx),
                            conv_cache_tensor.clone(),
                        ));
                    }
                }
            }
        }

        safetensors::tensor::serialize_to_file(data, Some(metadata), cache_file_path.as_path())
            .map_err(|e| LociError::Cache(e.to_string()))?;

        info!("Cache saved to {}", cache_file_path.display());

        Ok(cache_file_path)
    }

    fn enforce_limits(&self, cache_meta: &mut Vec<CacheMetadata>) -> Result<(), LociError> {
        let read_dir = fs::read_dir(&self.cache_dir).map_err(|e| LociError::IoWithContext {
            context: "failed to read cache directory",
            source: e,
        })?;
        let (lower_bound, _) = read_dir.size_hint();
        let mut file_metadata = Vec::with_capacity(lower_bound);
        let mut current_total_size = 0;
        for dir_entry in read_dir {
            match get_file_meta(dir_entry, &self.model, false, None) {
                Ok((Some(file_meta), _)) => {
                    current_total_size += file_meta.file_size;
                    file_metadata.push(file_meta);
                }
                Ok(_) => continue,
                Err(e) => error!("{}", e),
            }
        }

        evict_cache(
            &mut file_metadata,
            cache_meta,
            current_total_size,
            self.max_cache_size,
        )?;

        Ok(())
    }
}

impl ModelCacheManager {
    pub fn builder(model_name: &str) -> ModelCacheManagerBuilder {
        ModelCacheManagerBuilder::new(model_name)
    }

    pub fn load_cache_metadata(
        &self,
        fragmentation: &ModelCacheFragmentation,
    ) -> Result<Vec<CacheMetadata>, LociError> {
        let read_dir = fs::read_dir(&self.cache_dir).map_err(|e| LociError::IoWithContext {
            context: "failed to read cache directory",
            source: e,
        })?;
        let (lower_bound, _) = read_dir.size_hint();
        let cache_block_size = match fragmentation {
            ModelCacheFragmentation::BlockWise { block_size } => *block_size,
            _ => 1,
        };
        let mut cache_metadata = Vec::with_capacity(lower_bound);
        let mut file_metadata = Vec::with_capacity(lower_bound);
        let mut current_total_size = 0;
        for dir_entry in read_dir {
            match get_file_meta(dir_entry, &self.model, true, Some(cache_block_size)) {
                Ok((Some(file_meta), Some(cache))) => {
                    cache_metadata.push(cache);
                    current_total_size += file_meta.file_size;
                    file_metadata.push(file_meta);
                }
                Ok((Some(file_meta), None)) => {
                    current_total_size += file_meta.file_size;
                    file_metadata.push(file_meta);
                }
                Ok((_, _)) => continue,
                Err(e) => error!("{}", e),
            }
        }

        evict_cache(
            &mut file_metadata,
            &mut cache_metadata,
            current_total_size,
            self.max_cache_size,
        )?;

        info!(
            "Loaded {} cache files for model {}",
            cache_metadata.len(),
            &self.model
        );

        Ok(cache_metadata)
    }
}

#[cfg_attr(test, automock)]
pub trait CacheLoader {
    fn load_mixed_cache(
        &self,
        cache_file_path: PathBuf,
        matched_token_length: usize,
        fragmentation: &ModelCacheFragmentation,
        cache_seq_len_dim: usize,
        device: &Device,
        conv_on_cpu: bool,
    ) -> anyhow::Result<LoadedMixedCache>;
}

pub struct FileCacheLoader;

impl CacheLoader for FileCacheLoader {
    fn load_mixed_cache(
        &self,
        cache_file_path: PathBuf,
        matched_token_length: usize,
        fragmentation: &ModelCacheFragmentation,
        cache_seq_len_dim: usize,
        device: &Device,
        conv_on_cpu: bool,
    ) -> anyhow::Result<LoadedMixedCache> {
        let file = File::open(cache_file_path)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let (_, metadata) = SafeTensors::read_metadata(&mmap)?;
        let safetensors = SafeTensors::deserialize(&mmap)?;

        let max_layer = if let Some(n_layers) = metadata
            .metadata()
            .as_ref()
            .and_then(|meta| meta.get("n_layers"))
            .and_then(|n_layers_str| n_layers_str.parse::<usize>().ok())
        {
            n_layers
        } else {
            metadata
                .offset_keys()
                .iter()
                .filter_map(|k| k.strip_prefix("layer_"))
                .filter_map(|k| k.split('_').next()?.parse::<usize>().ok())
                .max()
                .unwrap_or(0)
        };

        let conv_device = if conv_on_cpu && !device.is_cpu() {
            &Device::Cpu
        } else {
            device
        };

        let (mut block_boundary_conv_cache, active_conv_block_idx, matched_token_length) =
            match fragmentation {
                ModelCacheFragmentation::BlockWise { block_size } => {
                    let meta_block_size = metadata
                        .metadata()
                        .as_ref()
                        .and_then(|meta| meta.get("block_size"))
                        .and_then(|block_size_str| block_size_str.parse::<usize>().ok())
                        .unwrap_or(1);
                    if meta_block_size != *block_size {
                        return Err(anyhow::anyhow!(format!(
                            "Cache block size ({}) does not match requested block size ({})",
                            meta_block_size, block_size
                        )));
                    }

                    let block_count = matched_token_length / (*block_size);
                    let block_boundary_conv_cache = vec![vec![None; max_layer]; block_count];
                    let active_conv_block_idx = block_count.checked_sub(1);
                    (
                        block_boundary_conv_cache,
                        active_conv_block_idx,
                        block_count * (*block_size),
                    )
                }
                ModelCacheFragmentation::TokenWise => (Vec::new(), None, matched_token_length),
            };

        let mut cache = vec![None; max_layer];
        for i in 0..=max_layer {
            let k_tensor = safetensors.tensor(&format!("layer_{}_k", i)).ok();
            let v_tensor = safetensors.tensor(&format!("layer_{}_v", i)).ok();
            if let (Some(k), Some(v)) = (k_tensor, v_tensor) {
                let k_raw = Tensor::from_raw_buffer(
                    k.data(),
                    get_candle_dtype(k.dtype())?,
                    k.shape(),
                    device,
                )?;
                let k_narrow = k_raw.narrow(cache_seq_len_dim, 0, matched_token_length)?;
                let v_raw = Tensor::from_raw_buffer(
                    v.data(),
                    get_candle_dtype(v.dtype())?,
                    v.shape(),
                    device,
                )?;
                let v_narrow = v_raw.narrow(cache_seq_len_dim, 0, matched_token_length)?;

                let mut concat_kv_cache = ConcatKvCache::new(cache_seq_len_dim);
                _ = concat_kv_cache.append(&k_narrow, &v_narrow);
                cache[i] = Some(MixedCache::KvCache(concat_kv_cache));
                continue;
            }
            if let Some(conv_block_idx) = active_conv_block_idx {
                for (block_idx, entry) in block_boundary_conv_cache
                    .iter_mut()
                    .enumerate()
                    .take(conv_block_idx + 1)
                {
                    if let Ok(tensor) =
                        safetensors.tensor(&format!("block_{}_layer_{}_conv_cache", block_idx, i))
                    {
                        let conv_cache_tensor = Tensor::from_raw_buffer(
                            tensor.data(),
                            get_candle_dtype(tensor.dtype())?,
                            tensor.shape(),
                            conv_device,
                        )?;
                        entry[i] = Some(MixedCache::ConvCache(conv_cache_tensor.clone()));
                        if block_idx == conv_block_idx {
                            cache[i] = Some(MixedCache::ConvCache(conv_cache_tensor));
                        }
                    }
                }
            }
        }

        Ok(LoadedMixedCache {
            mixed_cache: cache,
            block_boundary_conv_cache,
            cached_token_length: matched_token_length,
        })
    }
}

fn get_file_meta(
    dir_entry: std::io::Result<fs::DirEntry>,
    model_name: &str,
    with_cache_meta: bool,
    cache_block_size: Option<usize>,
) -> anyhow::Result<(Option<CacheFileMetadata>, Option<CacheMetadata>)> {
    let dir_entry = dir_entry?;
    let metadata = dir_entry.metadata()?;
    let mut cache_meta_file = None;
    let mut cache_meta = None;
    let entry = dir_entry;
    if metadata.is_file()
        && entry
            .path()
            .extension()
            .is_some_and(|ext| ext == "safetensors")
    {
        cache_meta_file = Some(CacheFileMetadata {
            cache_file_path: entry.path(),
            modified: metadata.modified()?,
            file_size: metadata.len(),
        });
        if with_cache_meta {
            let file = File::open(entry.path())?;
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
            let (_, safetensors_meta) = SafeTensors::read_metadata(&mmap)?;

            if let Some(metadata) = safetensors_meta.metadata().as_ref() {
                let metadata_model_name = metadata.get("model_name");
                let metadata_block_size = metadata
                    .get("block_size")
                    .and_then(|block_size_str| block_size_str.parse::<usize>().ok())
                    .unwrap_or(1);
                if metadata_model_name
                    .map(|m| m.as_str() == model_name)
                    .unwrap_or(false)
                    && metadata_block_size == cache_block_size.unwrap_or(1)
                {
                    let token_ids = read_token_ids(&mmap)?;
                    cache_meta = Some(CacheMetadata {
                        model: model_name.to_string(),
                        token_ids,
                        cache_file_path: entry.path(),
                    })
                }
            }
        }
    }
    Ok((cache_meta_file, cache_meta))
}

fn evict_cache(
    files: &mut [CacheFileMetadata],
    cache_meta: &mut Vec<CacheMetadata>,
    mut current_total_size: u64,
    max_cache_size: u64,
) -> Result<(), LociError> {
    if current_total_size <= max_cache_size {
        return Ok(());
    }

    files.sort_unstable_by_key(|file_meta| file_meta.modified);
    let mut evicted_count = 0;
    for file_meta_entry in files.iter() {
        current_total_size = current_total_size.saturating_sub(file_meta_entry.file_size);
        let file_to_remove = &file_meta_entry.cache_file_path;
        fs::remove_file(file_to_remove).map_err(|_| {
            LociError::Cache(format!(
                "Failed to remove file {}",
                file_to_remove.display()
            ))
        })?;
        debug!("Evicted cache file {}", file_to_remove.display());
        evicted_count += 1;
        if current_total_size <= max_cache_size {
            break;
        }
    }

    let evicted_files = &files[..evicted_count];
    cache_meta.retain(|cache| {
        evicted_files
            .iter()
            .all(|file_meta| file_meta.cache_file_path != cache.cache_file_path)
    });

    Ok(())
}

fn read_token_ids(mmap: &memmap2::Mmap) -> Result<Vec<u32>, LociError> {
    let safetensors =
        SafeTensors::deserialize(mmap).map_err(|e| LociError::CacheLoad(e.to_string()))?;
    let tensor = safetensors
        .tensor("token_ids")
        .map_err(|e| LociError::CacheLoad(format!("token_ids not found: {}", e)))?;

    if tensor.dtype() != safetensors::Dtype::U32 {
        return Err(LociError::CacheLoad("token_ids must be u32".to_string()));
    }

    let shape = tensor.shape();
    if shape.len() != 1 {
        return Err(LociError::CacheLoad("token_ids must be 1d".to_string()));
    }
    let mut token_ids = vec![0u32; shape[0]];
    LittleEndian::read_u32_into(tensor.data(), &mut token_ids);

    Ok(token_ids)
}

fn get_candle_dtype(dtype: safetensors::Dtype) -> Result<DType, LociError> {
    match dtype {
        safetensors::Dtype::F16 => Ok(DType::F16),
        safetensors::Dtype::BF16 => Ok(DType::BF16),
        safetensors::Dtype::F32 => Ok(DType::F32),
        safetensors::Dtype::F64 => Ok(DType::F64),
        safetensors::Dtype::U8 => Ok(DType::U8),
        safetensors::Dtype::U32 => Ok(DType::U32),
        _ => Err(LociError::Cache(format!("Unsupported dtype: {:?}", dtype))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // A small helper struct to hold our setup data cleanly
    struct TestEnv {
        _dir: TempDir,              // Kept alive so the directory isn't deleted during the test
        manager: ModelCacheManager, // Replace with your actual struct name
    }

    fn setup_test_cache(min_tokens: usize, max_size: u64, model_name: &str) -> TestEnv {
        // 1. Create a secure, isolated temporary directory on the hard drive
        let tmp_dir = TempDir::new().unwrap();

        let manager = ModelCacheManager {
            cache_dir: tmp_dir.path().to_path_buf(),
            model: model_name.to_string(),
            max_cache_size: max_size,
            min_cache_tokens: min_tokens,
        };

        TestEnv {
            _dir: tmp_dir,
            manager,
        }
    }

    fn add_test_cache_file(
        cache_dir: PathBuf,
        block_size: usize,
        model_name: &str,
        n_layers: usize,
        fragmentation: ModelCacheFragmentation,
    ) {
        let cache_file_path = cache_dir.join(format!(
            "cache-{}-blk-size-{}-{}.safetensors",
            model_name,
            block_size,
            Uuid::new_v4()
        ));
        let mut metadata = HashMap::new();
        metadata.insert("model_name".to_string(), model_name.to_string());
        metadata.insert("n_layers".to_string(), n_layers.to_string());
        metadata.insert("fragmentation".to_string(), fragmentation.to_string());
        metadata.insert("block_size".to_string(), block_size.to_string());

        let mut data = Vec::new();
        let token_ids: Vec<u32> = vec![1, 2, 3];
        let token_ids_tensor = Tensor::from_slice(&token_ids, token_ids.len(), &Device::Cpu)
            .expect("token_ids_tensor creation expected to succeed, but failed");
        data.push(("token_ids".to_string(), token_ids_tensor));

        safetensors::tensor::serialize_to_file(data, Some(metadata), cache_file_path.as_path())
            .unwrap();
    }

    fn add_big_test_cache_file(
        cache_dir: PathBuf,
        block_size: usize,
        model_name: &str,
        n_layers: usize,
        fragmentation: ModelCacheFragmentation,
    ) {
        let cache_file_path = cache_dir.join(format!(
            "cache-{}-blk-size-{}-{}.safetensors",
            model_name,
            block_size,
            Uuid::new_v4()
        ));
        let mut metadata = HashMap::new();
        metadata.insert("model_name".to_string(), model_name.to_string());
        metadata.insert("n_layers".to_string(), n_layers.to_string());
        metadata.insert("fragmentation".to_string(), fragmentation.to_string());
        metadata.insert("block_size".to_string(), block_size.to_string());

        let mut data = Vec::new();
        let token_ids: Vec<u32> = vec![1, 2, 3];
        let token_ids_tensor = Tensor::from_slice(&token_ids, token_ids.len(), &Device::Cpu)
            .expect("token_ids_tensor creation expected to succeed, but failed");
        data.push(("token_ids".to_string(), token_ids_tensor));

        for i in 0..5000 {
            let dummy_tensor = Tensor::arange(0f64, 1024f64, &Device::Cpu)
                .expect("dummy_tensor creation expected to succeed, but failed");
            data.push((format!("dummy_tensor_{}", i), dummy_tensor));
        }

        safetensors::tensor::serialize_to_file(data, Some(metadata), cache_file_path.as_path())
            .unwrap();
    }

    #[test]
    fn test_load_cache_metadata_same_model() {
        let env = setup_test_cache(1, 10_000_000, "test_model");
        add_test_cache_file(
            env.manager.cache_dir.clone(),
            1,
            "test_model",
            2,
            ModelCacheFragmentation::TokenWise,
        );

        let metadata = env
            .manager
            .load_cache_metadata(&ModelCacheFragmentation::TokenWise)
            .expect("Failed to load cache metadata");
        assert!(!metadata.is_empty());
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata[0].model, "test_model");
        assert_eq!(metadata[0].token_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_load_cache_metadata_different_model() {
        let env = setup_test_cache(1, 10_000_000, "test_model");
        add_test_cache_file(
            env.manager.cache_dir.clone(),
            1,
            "other_model",
            2,
            ModelCacheFragmentation::TokenWise,
        );

        let metadata = env
            .manager
            .load_cache_metadata(&ModelCacheFragmentation::TokenWise)
            .expect("Failed to load cache metadata");
        assert!(metadata.is_empty());
    }

    #[test]
    fn test_load_cache_metadata_triggers_eviction() {
        let env = setup_test_cache(1, 1, "test_model");
        add_test_cache_file(
            env.manager.cache_dir.clone(),
            1,
            "test_model",
            2,
            ModelCacheFragmentation::TokenWise,
        );
        add_test_cache_file(
            env.manager.cache_dir.clone(),
            1,
            "test_model",
            2,
            ModelCacheFragmentation::TokenWise,
        );

        let metadata = env
            .manager
            .load_cache_metadata(&ModelCacheFragmentation::TokenWise)
            .expect("Failed to load cache metadata");
        assert!(metadata.is_empty());
        let read_dir = fs::read_dir(&env.manager.cache_dir).unwrap();
        assert_eq!(read_dir.count(), 0);
    }

    #[test]
    fn test_save_load_roundtrip_tokenwise_full() {
        let mut mixed_cache = vec![];
        let mut kv_layer_cache = ConcatKvCache::new(2);
        let k_tensor = Tensor::arange(0f64, 10f64, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let v_tensor = Tensor::arange(0f64, 10f64, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        kv_layer_cache
            .append(&k_tensor, &v_tensor)
            .expect("cache append failed");
        mixed_cache.push(Some(MixedCache::KvCache(kv_layer_cache)));
        let conv_layer_cache = MixedCache::ConvCache(
            Tensor::arange(10f64, 20f64, &Device::Cpu)
                .expect("conv_cache tensor creation failed")
                .unsqueeze(0)
                .unwrap()
                .unsqueeze(0)
                .unwrap(),
        );
        mixed_cache.push(Some(conv_layer_cache));

        let env = setup_test_cache(1, 10_000_000, "test_model");
        let file_path = env
            .manager
            .save_cache(
                &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                &mixed_cache,
                2,
                &ModelCacheFragmentation::TokenWise,
                &vec![],
            )
            .expect("cache save failed");
        let cache_loader = FileCacheLoader;
        let loaded_cache = cache_loader
            .load_mixed_cache(
                file_path.clone(),
                10,
                &ModelCacheFragmentation::TokenWise,
                2,
                &Device::Cpu,
                true,
            )
            .expect("load_mixed_cache failed");

        assert!(loaded_cache.block_boundary_conv_cache.is_empty());
        assert_eq!(loaded_cache.cached_token_length, 10);
        let loaded_mixed_cache = loaded_cache.mixed_cache;
        for layer_cache in loaded_mixed_cache.iter() {
            if let Some(MixedCache::KvCache(kv_cache)) = layer_cache {
                let k = kv_cache
                    .k()
                    .expect("getting k tensor from layer_cache failed");
                let v = kv_cache
                    .v()
                    .expect("getting v tensor from layer_cache failed");
                let k_actual = k.flatten_all().unwrap().to_vec1::<f64>().unwrap();
                let v_actual = v.flatten_all().unwrap().to_vec1::<f64>().unwrap();
                let k_expected = k_tensor.flatten_all().unwrap().to_vec1::<f64>().unwrap();
                let v_expected = v_tensor.flatten_all().unwrap().to_vec1::<f64>().unwrap();
                assert_eq!(k_actual, k_expected);
                assert_eq!(v_actual, v_expected);
            }
            if let Some(MixedCache::ConvCache(..)) = layer_cache {
                assert!(false, "conv cache should not be present");
            }
        }
    }

    #[test]
    fn test_save_load_roundtrip_tokenwise_truncated() {
        let mut mixed_cache = vec![];
        let mut kv_layer_cache = ConcatKvCache::new(2);
        let k_tensor = Tensor::arange(0f64, 10f64, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let v_tensor = Tensor::arange(0f64, 10f64, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        kv_layer_cache
            .append(&k_tensor, &v_tensor)
            .expect("cache append failed");
        mixed_cache.push(Some(MixedCache::KvCache(kv_layer_cache)));
        let conv_layer_cache = MixedCache::ConvCache(
            Tensor::arange(10f64, 20f64, &Device::Cpu)
                .expect("conv_cache tensor creation failed")
                .unsqueeze(0)
                .unwrap()
                .unsqueeze(0)
                .unwrap(),
        );
        mixed_cache.push(Some(conv_layer_cache));

        let env = setup_test_cache(1, 10_000_000, "test_model");
        let file_path = env
            .manager
            .save_cache(
                &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                &mixed_cache,
                2,
                &ModelCacheFragmentation::TokenWise,
                &vec![],
            )
            .expect("cache save failed");
        let cache_loader = FileCacheLoader;
        let loaded_cache = cache_loader
            .load_mixed_cache(
                file_path.clone(),
                7,
                &ModelCacheFragmentation::TokenWise,
                2,
                &Device::Cpu,
                true,
            )
            .expect("load_mixed_cache failed");

        assert!(loaded_cache.block_boundary_conv_cache.is_empty());
        assert_eq!(loaded_cache.cached_token_length, 7);
        let loaded_mixed_cache = loaded_cache.mixed_cache;
        for layer_cache in loaded_mixed_cache.iter() {
            if let Some(MixedCache::KvCache(kv_cache)) = layer_cache {
                let k = kv_cache
                    .k()
                    .expect("getting k tensor from layer_cache failed");
                let v = kv_cache
                    .v()
                    .expect("getting v tensor from layer_cache failed");
                let k_actual = k.flatten_all().unwrap().to_vec1::<f64>().unwrap();
                let v_actual = v.flatten_all().unwrap().to_vec1::<f64>().unwrap();
                let mut k_expected = k_tensor.flatten_all().unwrap().to_vec1::<f64>().unwrap();
                let mut v_expected = v_tensor.flatten_all().unwrap().to_vec1::<f64>().unwrap();
                k_expected.truncate(7);
                v_expected.truncate(7);
                assert_eq!(k_actual, k_expected);
                assert_eq!(v_actual, v_expected);
            }
            if let Some(MixedCache::ConvCache(..)) = layer_cache {
                assert!(false, "conv cache should not be present");
            }
        }
    }

    #[test]
    fn test_save_load_roundtrip_blockwise_full() {
        let mut mixed_cache = vec![];
        let mut kv_layer_cache = ConcatKvCache::new(2);
        let k_tensor = Tensor::arange(0f64, 10f64, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let v_tensor = Tensor::arange(0f64, 10f64, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        kv_layer_cache
            .append(&k_tensor, &v_tensor)
            .expect("cache append failed");
        mixed_cache.push(Some(MixedCache::KvCache(kv_layer_cache)));
        let conv_layer_cache = MixedCache::ConvCache(
            Tensor::arange(10f64, 20f64, &Device::Cpu)
                .expect("conv_cache tensor creation failed")
                .unsqueeze(0)
                .unwrap()
                .unsqueeze(0)
                .unwrap(),
        );
        mixed_cache.push(Some(conv_layer_cache));

        let fragmentation = ModelCacheFragmentation::BlockWise { block_size: 3 };
        let blocks_len = 10 / 3;
        let mut block_boundary_cache = vec![vec![None, None]; blocks_len];
        for block_idx in 0..blocks_len {
            let conv_layer_cache = MixedCache::ConvCache(
                Tensor::arange(
                    (1 * block_idx) as f64,
                    ((1 * block_idx) + 10) as f64,
                    &Device::Cpu,
                )
                .expect("conv_cache tensor creation failed")
                .unsqueeze(0)
                .unwrap()
                .unsqueeze(0)
                .unwrap(),
            );
            block_boundary_cache[block_idx][1] = Some(conv_layer_cache);
        }

        let env = setup_test_cache(1, 10_000_000, "test_model");
        let file_path = env
            .manager
            .save_cache(
                &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                &mixed_cache,
                2,
                &fragmentation,
                &block_boundary_cache,
            )
            .expect("cache save failed");
        let cache_loader = FileCacheLoader;
        let loaded_cache = cache_loader
            .load_mixed_cache(file_path.clone(), 10, &fragmentation, 2, &Device::Cpu, true)
            .expect("load_mixed_cache failed");

        assert_eq!(loaded_cache.cached_token_length, 9);
        let loaded_mixed_cache = loaded_cache.mixed_cache;
        for layer_cache in loaded_mixed_cache.iter() {
            if let Some(MixedCache::KvCache(kv_cache)) = layer_cache {
                let k = kv_cache
                    .k()
                    .expect("getting k tensor from layer_cache failed");
                let v = kv_cache
                    .v()
                    .expect("getting v tensor from layer_cache failed");
                let k_actual = k.flatten_all().unwrap().to_vec1::<f64>().unwrap();
                let v_actual = v.flatten_all().unwrap().to_vec1::<f64>().unwrap();
                let mut k_expected = k_tensor.flatten_all().unwrap().to_vec1::<f64>().unwrap();
                let mut v_expected = v_tensor.flatten_all().unwrap().to_vec1::<f64>().unwrap();
                k_expected.truncate(9);
                v_expected.truncate(9);
                assert_eq!(k_actual, k_expected);
                assert_eq!(v_actual, v_expected);
            }
            if let Some(MixedCache::ConvCache(conv_cache)) = layer_cache {
                let conv_cache_actual = conv_cache.flatten_all().unwrap().to_vec1::<f64>().unwrap();
                let block_boundary_layer_cache = block_boundary_cache.last().unwrap()[1]
                    .clone()
                    .expect("block boundary cache should be present");
                let block_boundary_conv_cache = block_boundary_layer_cache
                    .as_conv_cache()
                    .expect("block boundary cache should be present");
                let conv_cache_expected = block_boundary_conv_cache
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f64>()
                    .unwrap();
                assert_eq!(conv_cache_actual, conv_cache_expected);
            }
        }

        for block_idx in 0..blocks_len {
            let conv_layer_cache = loaded_cache.block_boundary_conv_cache[block_idx][1]
                .clone()
                .expect("loaded block boundary cache should be present");
            let conv_cache = conv_layer_cache
                .as_conv_cache()
                .expect("conv cache should be present");
            let conv_cache_actual = conv_cache.flatten_all().unwrap().to_vec1::<f64>().unwrap();
            let block_boundary_layer_cache = block_boundary_cache[block_idx][1]
                .clone()
                .expect("block boundary cache should be present");
            let block_boundary_conv_cache = block_boundary_layer_cache
                .as_conv_cache()
                .expect("block boundary cache should be present");
            let conv_cache_expected = block_boundary_conv_cache
                .flatten_all()
                .unwrap()
                .to_vec1::<f64>()
                .unwrap();
            assert_eq!(conv_cache_actual, conv_cache_expected);
        }
    }

    #[test]
    fn test_save_load_roundtrip_blockwise_truncated() {
        let mut mixed_cache = vec![];
        let mut kv_layer_cache = ConcatKvCache::new(2);
        let k_tensor = Tensor::arange(0f64, 10f64, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let v_tensor = Tensor::arange(0f64, 10f64, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        kv_layer_cache
            .append(&k_tensor, &v_tensor)
            .expect("cache append failed");
        mixed_cache.push(Some(MixedCache::KvCache(kv_layer_cache)));
        let conv_layer_cache = MixedCache::ConvCache(
            Tensor::arange(10f64, 20f64, &Device::Cpu)
                .expect("conv_cache tensor creation failed")
                .unsqueeze(0)
                .unwrap()
                .unsqueeze(0)
                .unwrap(),
        );
        mixed_cache.push(Some(conv_layer_cache));

        let fragmentation = ModelCacheFragmentation::BlockWise { block_size: 3 };
        let blocks_len = 10 / 3;
        let mut block_boundary_cache = vec![vec![None, None]; blocks_len];
        for block_idx in 0..blocks_len {
            let conv_layer_cache = MixedCache::ConvCache(
                Tensor::arange(
                    (1 * block_idx) as f64,
                    ((1 * block_idx) + 10) as f64,
                    &Device::Cpu,
                )
                .expect("conv_cache tensor creation failed")
                .unsqueeze(0)
                .unwrap()
                .unsqueeze(0)
                .unwrap(),
            );
            block_boundary_cache[block_idx][1] = Some(conv_layer_cache);
        }

        let env = setup_test_cache(1, 10_000_000, "test_model");
        let file_path = env
            .manager
            .save_cache(
                &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                &mixed_cache,
                2,
                &fragmentation,
                &block_boundary_cache,
            )
            .expect("cache save failed");
        let cache_loader = FileCacheLoader;
        let loaded_cache = cache_loader
            .load_mixed_cache(file_path.clone(), 7, &fragmentation, 2, &Device::Cpu, true)
            .expect("load_mixed_cache failed");

        assert_eq!(loaded_cache.cached_token_length, 6);
        let loaded_mixed_cache = loaded_cache.mixed_cache;
        for layer_cache in loaded_mixed_cache.iter() {
            if let Some(MixedCache::KvCache(kv_cache)) = layer_cache {
                let k = kv_cache
                    .k()
                    .expect("getting k tensor from layer_cache failed");
                let v = kv_cache
                    .v()
                    .expect("getting v tensor from layer_cache failed");
                let k_actual = k.flatten_all().unwrap().to_vec1::<f64>().unwrap();
                let v_actual = v.flatten_all().unwrap().to_vec1::<f64>().unwrap();
                let mut k_expected = k_tensor.flatten_all().unwrap().to_vec1::<f64>().unwrap();
                let mut v_expected = v_tensor.flatten_all().unwrap().to_vec1::<f64>().unwrap();
                k_expected.truncate(6);
                v_expected.truncate(6);
                assert_eq!(k_actual, k_expected);
                assert_eq!(v_actual, v_expected);
            }
            if let Some(MixedCache::ConvCache(conv_cache)) = layer_cache {
                let conv_cache_actual = conv_cache.flatten_all().unwrap().to_vec1::<f64>().unwrap();
                let block_boundary_layer_cache = block_boundary_cache[1][1]
                    .clone()
                    .expect("block boundary cache should be present");
                let block_boundary_conv_cache = block_boundary_layer_cache
                    .as_conv_cache()
                    .expect("block boundary cache should be present");
                let conv_cache_expected = block_boundary_conv_cache
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f64>()
                    .unwrap();
                assert_eq!(conv_cache_actual, conv_cache_expected);
            }
        }

        for block_idx in 0..(blocks_len - 1) {
            let conv_layer_cache = loaded_cache.block_boundary_conv_cache[block_idx][1]
                .clone()
                .expect("loaded block boundary cache should be present");
            let conv_cache = conv_layer_cache
                .as_conv_cache()
                .expect("conv cache should be present");
            let conv_cache_actual = conv_cache.flatten_all().unwrap().to_vec1::<f64>().unwrap();
            let block_boundary_layer_cache = block_boundary_cache[block_idx][1]
                .clone()
                .expect("block boundary cache should be present");
            let block_boundary_conv_cache = block_boundary_layer_cache
                .as_conv_cache()
                .expect("block boundary cache should be present");
            let conv_cache_expected = block_boundary_conv_cache
                .flatten_all()
                .unwrap()
                .to_vec1::<f64>()
                .unwrap();
            assert_eq!(conv_cache_actual, conv_cache_expected);
        }
        assert_eq!(loaded_cache.block_boundary_conv_cache.len(), 2);
    }

    #[test]
    fn test_enforce_limits() {
        let mut env = setup_test_cache(1, 1024_000_000, "test_model");
        add_big_test_cache_file(
            env.manager.cache_dir.clone(),
            1,
            "different_test_model",
            2,
            ModelCacheFragmentation::TokenWise,
        );
        // ensure stable ordering: small file must be strictly newer
        std::thread::sleep(std::time::Duration::from_millis(10));
        add_test_cache_file(
            env.manager.cache_dir.clone(),
            1,
            "test_model",
            2,
            ModelCacheFragmentation::TokenWise,
        );

        let metadata = env
            .manager
            .load_cache_metadata(&ModelCacheFragmentation::TokenWise)
            .expect("Failed to load cache metadata");
        assert!(
            !metadata.is_empty(),
            "'test_model' cache metadata should be present before enforcing limits"
        );
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata[0].model, "test_model");

        env.manager.model = "different_test_model".to_string();
        let mut metadata = env
            .manager
            .load_cache_metadata(&ModelCacheFragmentation::TokenWise)
            .expect("Failed to load cache metadata");
        assert!(
            !metadata.is_empty(),
            "'different_test_model' cache metadata should be present"
        );
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata[0].model, "different_test_model");

        env.manager.max_cache_size = 1_000_000;

        env.manager
            .enforce_limits(&mut metadata)
            .expect("enforce limits failed");
        assert!(metadata.is_empty());

        let read_dir = fs::read_dir(&env.manager.cache_dir)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(read_dir.len(), 1);

        for entry in read_dir {
            let path = entry.path();
            let file = File::open(path).unwrap();
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
            let (_, safetensors_meta) = SafeTensors::read_metadata(&mmap).unwrap();
            let metadata = safetensors_meta.metadata().as_ref().unwrap();
            let model_name = metadata.get("model_name").unwrap();
            assert_eq!(model_name, "test_model");
        }
    }

    #[test]
    fn test_read_token_ids_success() {
        let temp_dir = TempDir::new().expect("temp dir failed");
        let file_path = temp_dir.path().join("test_token_ids.safetensors");
        let token_ids = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let data = vec![(
            "token_ids".to_string(),
            Tensor::from_slice(&token_ids, token_ids.len(), &Device::Cpu).unwrap(),
        )];
        safetensors::tensor::serialize_to_file(data, None, file_path.as_path())
            .expect("safetensors serialization failed");

        let file = File::open(file_path).unwrap();
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
        let token_ids = read_token_ids(&mmap).unwrap();
        assert_eq!(token_ids, token_ids);
    }

    #[rstest::rstest]
    #[case("not_token_ids", Tensor::from_slice(&[1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10, &Device::Cpu).unwrap(), "token_ids not found")]
    #[case("token_ids", Tensor::from_slice(&[1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.], 10, &Device::Cpu).unwrap(), "token_ids must be u32")]
    #[case("token_ids", Tensor::from_slice(&[1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10], &[1, 10], &Device::Cpu).unwrap(), "token_ids must be 1d")]
    fn test_read_token_ids_failure(
        #[case] token_ids_tensor_name: &str,
        #[case] token_ids_tensor: Tensor,
        #[case] expected_error: &str,
    ) {
        let temp_dir = TempDir::new().expect("temp dir failed");
        let file_path = temp_dir.path().join("test_token_ids.safetensors");
        let data = vec![(token_ids_tensor_name.to_string(), token_ids_tensor)];
        safetensors::tensor::serialize_to_file(data, None, file_path.as_path())
            .expect("safetensors serialization failed");

        let file = File::open(file_path).unwrap();
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
        let error =
            read_token_ids(&mmap).expect_err("read_token_ids expected to fail, but succeeded");
        assert!(error.to_string().contains(expected_error));
    }

    #[rstest::rstest]
    #[case(safetensors::Dtype::U32, DType::U32)]
    #[case(safetensors::Dtype::F16, DType::F16)]
    #[case(safetensors::Dtype::BF16, DType::BF16)]
    #[case(safetensors::Dtype::F32, DType::F32)]
    #[case(safetensors::Dtype::F64, DType::F64)]
    #[case(safetensors::Dtype::U8, DType::U8)]
    fn test_get_candle_dtype_success(
        #[case] safetensors_dtype: safetensors::Dtype,
        #[case] expected_dtype: DType,
    ) {
        let dtype = get_candle_dtype(safetensors_dtype).expect("get_candle_dtype failed");
        assert_eq!(dtype, expected_dtype);
    }

    #[test]
    fn test_get_candle_dtype_failure() {
        let error = get_candle_dtype(safetensors::Dtype::U16)
            .expect_err("get_candle_dtype expected to fail, but succeeded");
        assert!(error.to_string().contains("Unsupported dtype"));
    }
}
