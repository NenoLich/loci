use crate::config::ModelCacheConfig;
use crate::error::LociError;
use crate::inference::GenerationContext;
use crate::model::MixedCache;
use crate::types::ModelCacheFragmentation;
use std::collections::{HashMap, HashSet};
use anyhow::Context;
use std::fs;
use std::fs::File;
use std::path::{PathBuf, Path};
use std::io::Cursor;
use byteorder::{LittleEndian, ByteOrder};
use candle_core::{Tensor, Device, DType};
use candle_nn::kv_cache::ConcatKvCache;
use safetensors::SafeTensors;
use uuid::Uuid;
use tracing::{error, info, debug};

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
        Self { config: None, model: model.to_string() }
    }

    pub fn with_config(mut self, config: ModelCacheConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn build(self) -> ModelCacheManager {
        let config = if let Some(config) = self.config {
            config
        } else {
            ModelCacheConfig::default()
        };
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

#[derive(Default)]
pub struct ModelCacheManager {
    pub model: String,
    pub cache_dir: PathBuf,
    pub max_cache_size: u64,
    pub min_cache_tokens: usize,
}

impl ModelCacheManager {
    pub fn builder(model_name: &str) -> ModelCacheManagerBuilder {
        ModelCacheManagerBuilder::new(model_name)
    }

    pub fn save_cache(&self, token_ids: &[u32], cache: &Vec<Option<MixedCache>>, cache_seq_len_dim: usize, fragmentation: &ModelCacheFragmentation, block_boundary_conv_cache: &Vec<Vec<Option<MixedCache>>>) -> Result<PathBuf, LociError> {
        let (mut data, token_len_to_save, block_size) = match fragmentation {
            ModelCacheFragmentation::BlockWise { block_size } => 
                (Vec::with_capacity(cache.len() * 2 + 1 + (block_boundary_conv_cache.len() * cache.len())), token_ids.len() / (*block_size) * (*block_size), *block_size),
            ModelCacheFragmentation::TokenWise => (Vec::with_capacity(cache.len() * 2 + 1), token_ids.len(), 1),
        };
        let cache_file_path = self.cache_dir.join(format!("cache-{}-blk-size-{}-{}.safetensors", &self.model, block_size, Uuid::new_v4()));

        let mut metadata = HashMap::new();
        metadata.insert("model_name".to_string(), self.model.clone());
        metadata.insert("n_layers".to_string(), cache.len().to_string());
        metadata.insert("fragmentation".to_string(), fragmentation.to_string());
        metadata.insert("block_size".to_string(), block_size.to_string());

        let token_ids_tensor = Tensor::from_slice(&token_ids[..token_len_to_save], (token_len_to_save), &Device::Cpu)
            .map_err(|e| LociError::Cache(format!("failed to create token_ids tensor: {}", e)))?;
        data.push(("token_ids".to_string(), token_ids_tensor));
        for (i, layer) in cache.iter().enumerate() {
            match layer.as_ref() {
                Some(MixedCache::KvCache(concat_kv_cache)) => {
                    match (concat_kv_cache.k(), concat_kv_cache.v()) {
                        (Some(k), Some(v)) => {
                            let k_to_save = k.narrow(cache_seq_len_dim, 0, token_len_to_save)
                                .map_err(|e| LociError::Cache(e.to_string()))?;
                            let v_to_save = v.narrow(cache_seq_len_dim, 0, token_len_to_save)
                                .map_err(|e| LociError::Cache(e.to_string()))?;
                            data.push((format!("layer_{}_k", i), k_to_save));
                            data.push((format!("layer_{}_v", i), v_to_save));
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        if let ModelCacheFragmentation::BlockWise { .. } = fragmentation {
            for (block_idx, block_boundary_cache) in block_boundary_conv_cache.iter().enumerate() {
                for (layer_idx, layer_cache) in block_boundary_cache.iter().enumerate() {
                    match layer_cache.as_ref() {
                        Some(MixedCache::ConvCache(conv_cache_tensor)) => {
                            data.push((format!("block_{}_layer_{}_conv_cache", block_idx, layer_idx), conv_cache_tensor.clone()));
                        }
                        _ => {}
                    }
                }
            }
        }

        safetensors::tensor::serialize_to_file(data, Some(metadata), cache_file_path.as_path())
            .map_err(|e| LociError::Cache(e.to_string()))?;

        info!("Cache saved to {}", cache_file_path.display());

        Ok(cache_file_path)
    }

    pub fn enforce_limits(&self, cache_meta: &mut Vec<CacheMetadata>) -> Result<(), LociError> {
        let read_dir = fs::read_dir(&self.cache_dir)
            .map_err(|e| LociError::IoWithContext { context: "failed to read cache directory", source: e })?;
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

        evict_cache(&mut file_metadata, cache_meta, current_total_size, self.max_cache_size)?;

        Ok(())
    }

    pub fn load_cache_metadata(&self, fragmentation: &ModelCacheFragmentation) -> Result<Vec<CacheMetadata>, LociError> {
        let read_dir = fs::read_dir(&self.cache_dir)
            .map_err(|e| LociError::IoWithContext { context: "failed to read cache directory", source: e })?;
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
                },
                Ok((_, _)) => continue,
                Err(e) => error!("{}", e),
            }
        }

        evict_cache(&mut file_metadata, &mut cache_metadata, current_total_size, self.max_cache_size)?;

        info!("Loaded {} cache files for model {}", cache_metadata.len(), &self.model);

        Ok(cache_metadata)
    }
}

fn get_file_meta(dir_entry: std::io::Result<fs::DirEntry>, model_name: &str, with_cache_meta: bool, cache_block_size: Option<usize>) -> anyhow::Result<(Option<CacheFileMetadata>, Option<CacheMetadata>)> {
    let dir_entry = dir_entry?;
    let metadata = dir_entry.metadata()?;
    let mut cache_meta_file = None;
    let mut cache_meta = None;
    let entry = dir_entry;
    if metadata.is_file() && entry.path().extension().map_or(false, |ext| ext == "safetensors") {
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
                let metadata_block_size = metadata.get("block_size")
                    .and_then(|block_size_str| block_size_str.parse::<usize>().ok())
                    .unwrap_or(1);
                if metadata_model_name.map(|m| m.as_str() == model_name).unwrap_or(false) 
                    && metadata_block_size == cache_block_size.unwrap_or(1) 
                    {
                    let safetensors = SafeTensors::deserialize(&mmap)?;
                    let token_ids = read_token_ids(&mmap, safetensors_meta)?;
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

fn evict_cache(files: &mut Vec<CacheFileMetadata>, cache_meta: &mut Vec<CacheMetadata>, mut current_total_size: u64, max_cache_size: u64) -> Result<(), LociError> {
    if current_total_size <= max_cache_size {
        return Ok(()); 
    }

    files.sort_unstable_by_key(|file_meta| file_meta.modified);
    let mut evicted_count = 0;
    for file_meta_entry in files.iter() {
        current_total_size = current_total_size.saturating_sub(file_meta_entry.file_size);
        let file_to_remove = &file_meta_entry.cache_file_path;
        fs::remove_file(file_to_remove).map_err(|_| LociError::Cache(format!("Failed to remove file {}", file_to_remove.display())))?;
        debug!("Evicted cache file {}", file_to_remove.display());
        evicted_count += 1;
        if current_total_size <= max_cache_size {
            break;
        }
    }

    let evicted_files = &files[..evicted_count];
    let files_to_remove = evicted_files.iter().map(|file_meta| file_meta.cache_file_path.clone()).collect::<Vec<_>>();
    cache_meta.retain(|cache| evicted_files.iter().all(|file_meta| file_meta.cache_file_path != cache.cache_file_path));

    Ok(())
}

fn read_token_ids(mmap: &memmap2::Mmap, metadata: safetensors::tensor::Metadata) -> Result<Vec<u32>, LociError> {
    let safetensors = SafeTensors::deserialize(mmap)
        .map_err(|e| LociError::CacheLoad(e.to_string()))?;
    let tensor = safetensors.tensor("token_ids")
        .map_err(|e| LociError::CacheLoad(format!("Failed to deserialize safetensors: {}", e)))?;

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

pub fn load_mixed_cache(cache_file_path: impl AsRef<Path>, cache_token_length: usize, matched_token_length: usize, fragmentation: &ModelCacheFragmentation, cache_seq_len_dim: usize, device: &Device, conv_on_cpu: bool) -> anyhow::Result<LoadedMixedCache> {
    let file = File::open(cache_file_path)?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let (_, metadata) = SafeTensors::read_metadata(&mmap)?;
    let safetensors = SafeTensors::deserialize(&mmap)?;

    let max_layer = if let Some(n_layers) = metadata.metadata()
        .as_ref()
        .and_then(|meta| meta.get("n_layers"))
        .and_then(|n_layers_str| n_layers_str.parse::<usize>().ok()) {
            n_layers
    } else {
        metadata.offset_keys().iter()
            .filter_map(|k| k.strip_prefix("layer_"))
            .filter_map(|k| k.split('_').next()?.parse::<usize>().ok())
            .max()
            .unwrap_or(0)
    };

    let conv_device = if conv_on_cpu && !device.is_cpu() {
        &Device::Cpu
    } else {
        &device
    };

    let (mut block_boundary_conv_cache, active_conv_block_idx, matched_token_length) = match fragmentation {
        ModelCacheFragmentation::BlockWise { block_size } => {
            let meta_block_size = metadata.metadata()
                .as_ref()
                .and_then(|meta| meta.get("block_size"))
                .and_then(|block_size_str| block_size_str.parse::<usize>().ok())
                .unwrap_or(1);
            if meta_block_size != *block_size {
                return Err(anyhow::anyhow!(format!("Cache block size ({}) does not match requested block size ({})", meta_block_size, block_size)));
            }

            let block_count = matched_token_length / (*block_size);
            let block_boundary_conv_cache = vec![vec![None; max_layer]; block_count];
            let active_conv_block_idx = block_count.checked_sub(1);
            (block_boundary_conv_cache, active_conv_block_idx, block_count * (*block_size))
        },
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
                device
            )?;
            let k_narrow = k_raw.narrow(cache_seq_len_dim, 0, matched_token_length)?;
            let v_raw = Tensor::from_raw_buffer(
                v.data(), 
                get_candle_dtype(v.dtype())?, 
                v.shape(), 
                device
            )?;
            let v_narrow = v_raw.narrow(cache_seq_len_dim, 0, matched_token_length)?;

            let mut concat_kv_cache = ConcatKvCache::new(cache_seq_len_dim);
            _ = concat_kv_cache.append(&k_narrow, &v_narrow);
            cache[i] = Some(MixedCache::KvCache(concat_kv_cache));
            continue;
        }
        if let Some(conv_block_idx) = active_conv_block_idx {
            for block_idx in 0..=conv_block_idx {
                if let Some(tensor) = safetensors.tensor(&format!("block_{}_layer_{}_conv_cache", block_idx, i)).ok() {
                    let conv_cache_tensor = Tensor::from_raw_buffer(
                        tensor.data(), 
                        get_candle_dtype(tensor.dtype())?, 
                        tensor.shape(), 
                        conv_device
                    )?;
                    block_boundary_conv_cache[block_idx][i] = Some(MixedCache::ConvCache(conv_cache_tensor.clone()));
                    if block_idx == conv_block_idx {
                        cache[i] = Some(MixedCache::ConvCache(conv_cache_tensor));
                        continue;
                    }
                }
            }
        }
    }

    Ok(LoadedMixedCache { mixed_cache: cache, block_boundary_conv_cache, cached_token_length: matched_token_length })
}