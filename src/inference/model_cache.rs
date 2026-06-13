use crate::config::ModelCacheConfig;
use crate::error::LociError;
use crate::inference::GenerationContext;
use crate::model::MixedCache;
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

#[derive(Debug)]
pub enum MatchCacheResult {
    FullyMatchedActiveCache,
    PartiallyMatchedActiveCache {
        matched_cache_length: usize,
    },
    MatchedInactiveCache {
        matched_cache_length: usize,
        cache: Vec<Option<MixedCache>>,
    },
    NoMatch,
}

struct CacheFileMetadata {
    cache_file_path: PathBuf,
    modified: std::time::SystemTime,
    file_size: u64,
}

pub struct CacheMetadata {
    pub model_name: String,
    pub token_ids: Vec<u32>,
    pub cache_file_path: PathBuf,
}

pub struct ModelCacheManagerBuilder {
    config: Option<ModelCacheConfig>,
    model_name: String,
    prefix_caching: bool,
}

impl ModelCacheManagerBuilder {
    pub fn new(model_name: &str) -> Self {
        Self { config: None, model_name: model_name.to_string(), prefix_caching: false }
    }

    pub fn prefix_caching(mut self, prefix_caching: bool) -> Self {
        self.prefix_caching = prefix_caching;
        self
    }

    pub fn with_config(mut self, config: ModelCacheConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn build(self) -> anyhow::Result<ModelCacheManager> {
        let config = if let Some(config) = self.config {
            config
        } else {
            ModelCacheConfig::default()
        };
        let cache_dir = config.cache_dir;
        let max_cache_size = config.max_cache_size;
        let min_cache_tokens = config.min_cache_tokens;

        let read_dir = fs::read_dir(&cache_dir)?;
        let (lower_bound, _) = read_dir.size_hint();
        let mut cache_metadata = Vec::with_capacity(lower_bound);
        let mut file_metadata = Vec::with_capacity(lower_bound);
        let mut current_total_size = 0;
        for dir_entry in read_dir {
            match get_file_meta_with_cache_meta(dir_entry, &self.model_name) {
                Ok((Some(cache), Some(file_meta))) => {
                    cache_metadata.push(cache);
                    current_total_size += file_meta.file_size;
                    file_metadata.push(file_meta);
                }
                Ok((None, Some(file_meta))) => {
                    current_total_size += file_meta.file_size;
                    file_metadata.push(file_meta);
                },
                Ok((_, _)) => continue,
                Err(e) => error!("{}", e),
            }
        }

        if current_total_size > max_cache_size {
            file_metadata.sort_unstable_by_key(|file_meta| file_meta.modified);
            let mut files_to_remove = HashSet::with_capacity(file_metadata.len());
            for file_meta_entry in file_metadata {
                current_total_size = current_total_size.saturating_sub(file_meta_entry.file_size);
                files_to_remove.insert(file_meta_entry.cache_file_path);
                if current_total_size <= max_cache_size {
                    break;
                }
            }

            cache_metadata.retain(|cache| !files_to_remove.contains(&cache.cache_file_path));

            for file_to_remove in files_to_remove {
                fs::remove_file(&file_to_remove).context(format!("Failed to remove file {}", file_to_remove.display()))?;
                debug!("Evicted cache file {}", file_to_remove.display());
            }
        }

        info!("Loaded {} cache files for model {}", cache_metadata.len(), &self.model_name);
 
        Ok(ModelCacheManager {
            model_name: self.model_name,
            cache_metadata,
            cache_dir,
            max_cache_size,
            min_cache_tokens,
        })
    }
}

#[derive(Default)]
pub struct ModelCacheManager {
    pub model_name: String,
    pub cache_metadata: Vec<CacheMetadata>,
    pub cache_dir: PathBuf,
    pub max_cache_size: u64,
    pub min_cache_tokens: usize,
}

impl ModelCacheManager {
    pub fn builder(model_name: &str) -> ModelCacheManagerBuilder {
        ModelCacheManagerBuilder::new(model_name)
    }

    pub fn match_cache(&mut self, ctx: &GenerationContext, input_token_ids: &[u32], cache_seq_len_dim: usize, min_prefill_tokens: usize, device: &Device) -> Result<MatchCacheResult, LociError> {
        let input_token_len = input_token_ids.len();
        let max_cache_len = input_token_len.saturating_sub(min_prefill_tokens);
        let mut highest_cache_match_len = ctx.token_ids.iter()
            .zip(input_token_ids.iter())
            .take_while(|(a, b)| a == b)
            .count();
        // Only return fully matched active cache if cache is not empty, input is not subslice of cache and cache is fully matched
        if highest_cache_match_len != 0 {
            // Avoid when input is fully overlap with cache, we need to pass at least one token to model forward

            if highest_cache_match_len > max_cache_len {        
                return Ok(MatchCacheResult::PartiallyMatchedActiveCache { 
                    matched_cache_length: max_cache_len, 
                });
            }
            if highest_cache_match_len == ctx.token_ids.len() {
                return Ok(MatchCacheResult::FullyMatchedActiveCache);
            }
        }

        let mut active_cache_source = true;
        let mut cache_len_with_highest_match = 0;
        let mut cache_file_path_with_highest_match = None;
        for cache_metadata in &self.cache_metadata {
            let cache_match_len = cache_metadata.token_ids.iter()
                .zip(input_token_ids.iter())
                .take_while(|(a, b)| a == b)
                .count();
            if cache_match_len > highest_cache_match_len && cache_match_len >= self.min_cache_tokens {
                highest_cache_match_len = cache_match_len;
                active_cache_source = false;
                cache_file_path_with_highest_match = Some(cache_metadata.cache_file_path.clone());
                cache_len_with_highest_match = cache_metadata.token_ids.len();
            }
        }
        // Avoid when input is fully overlap with cache, we need to pass at least one token to model forward
        highest_cache_match_len = highest_cache_match_len.min(max_cache_len);

        let result = match (highest_cache_match_len, active_cache_source, cache_file_path_with_highest_match) {
            (0, _, _) => MatchCacheResult::NoMatch,
            (_, true, _) => {
                self.save_cache(ctx)?;
                MatchCacheResult::PartiallyMatchedActiveCache { matched_cache_length: highest_cache_match_len }
            },
            (_, false, Some(cache_file_path)) => {
                self.save_cache(ctx)?;
                match self.load_cache(cache_file_path, cache_len_with_highest_match, highest_cache_match_len, cache_seq_len_dim, device)
                        .map_err(|e| LociError::CacheLoad(format!("failed to load cache: {}", e))) {
                            Ok(cache) => MatchCacheResult::MatchedInactiveCache { matched_cache_length: highest_cache_match_len, cache },
                            Err(e) => {
                                error!("{}", e);
                                MatchCacheResult::NoMatch
                            },
                        }
            },
            (_, false, None) => MatchCacheResult::NoMatch,
        };    

        Ok(result)
    }

    pub fn save_cache(&mut self, ctx: &GenerationContext) -> Result<(), LociError> {
        if ctx.token_ids.len() < self.min_cache_tokens {
            return Ok(());
        }
        let cache_file_path = self.cache_dir.join(format!("cache-{}-{}.safetensors", ctx.model_name, Uuid::new_v4()));

        let mut metadata = HashMap::new();
        metadata.insert("model_name".to_string(), ctx.model_name.clone());
        metadata.insert("n_layers".to_string(), ctx.cache.len().to_string());

        let mut data = Vec::with_capacity(ctx.cache.len() * 2 + 1);
        let token_ids_tensor = Tensor::from_slice(&ctx.token_ids, (ctx.token_ids.len()), &Device::Cpu)
            .map_err(|e| LociError::Cache(format!("failed to create token_ids tensor: {}", e)))?;
        data.push(("token_ids".to_string(), &token_ids_tensor));
        for (i, layer) in ctx.cache.iter().enumerate() {
            match layer.as_ref() {
                Some(MixedCache::KvCache(concat_kv_cache)) => {
                    match (concat_kv_cache.k(), concat_kv_cache.v()) {
                        (Some(k), Some(v)) => {
                            data.push((format!("layer_{}_k", i), k));
                            data.push((format!("layer_{}_v", i), v));
                        }
                        _ => {}
                    }
                }
                Some(MixedCache::ConvCache(conv_cache_tensor)) => {
                    data.push((format!("layer_{}_conv_cache", i), conv_cache_tensor));
                }
                None => {}
            }
        }
        safetensors::tensor::serialize_to_file(data, Some(metadata), cache_file_path.as_path())
            .map_err(|e| LociError::Cache(e.to_string()))?;

        info!("Cache saved to {}", cache_file_path.display());

        self.cache_metadata.push(CacheMetadata {
            model_name: ctx.model_name.clone(),
            token_ids: ctx.token_ids.clone(),
            cache_file_path,
        });

        self.enforce_limits()?;
        
        Ok(())
    }

    fn enforce_limits(&mut self) -> Result<(), LociError> {
        let read_dir = fs::read_dir(&self.cache_dir)?;
        let (lower_bound, _) = read_dir.size_hint();
        let mut file_metadata = Vec::with_capacity(lower_bound);
        let mut current_total_size = 0;
        for dir_entry in read_dir {
            match get_file_meta(dir_entry, &self.model_name) {
                Ok(Some(file_meta)) => {
                    current_total_size += file_meta.file_size;
                    file_metadata.push(file_meta);
                }
                Ok(None) => continue,
                Err(e) => error!("{}", e),
            }
        }

        if current_total_size > self.max_cache_size {
            file_metadata.sort_unstable_by_key(|file_meta| file_meta.modified);
            let mut files_to_remove = HashSet::with_capacity(file_metadata.len());
            for file_meta_entry in file_metadata {
                current_total_size = current_total_size.saturating_sub(file_meta_entry.file_size);
                files_to_remove.insert(file_meta_entry.cache_file_path);
                if current_total_size <= self.max_cache_size {
                    break;
                }
            }

            self.cache_metadata.retain(|cache| !files_to_remove.contains(&cache.cache_file_path));

            for file_to_remove in files_to_remove {
                fs::remove_file(&file_to_remove).map_err(|e| LociError::Cache(format!("Failed to remove file {}", file_to_remove.display())))?;
                debug!("Evicted cache file {}", file_to_remove.display());
            }
        }

        Ok(())
    }

    fn load_cache(&mut self, cache_file_path: impl AsRef<Path>, cache_token_lentgh: usize, matched_token_lentgh: usize, cache_seq_len_dim: usize, device: &Device) -> anyhow::Result<Vec<Option<MixedCache>>> {
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

        let mut cache = vec![None; max_layer];
        let cache_len_deficit = cache_token_lentgh - matched_token_lentgh;
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
                let k_narrow = k_raw.narrow(cache_seq_len_dim, 0, matched_token_lentgh)?;
                let v_raw = Tensor::from_raw_buffer(
                    v.data(), 
                    get_candle_dtype(v.dtype())?, 
                    v.shape(), 
                    device
                )?;
                let v_narrow = v_raw.narrow(cache_seq_len_dim, 0, matched_token_lentgh)?;

                let mut concat_kv_cache = ConcatKvCache::new(cache_seq_len_dim);
                _ = concat_kv_cache.append(&k_narrow, &v_narrow);
                cache[i] = Some(MixedCache::KvCache(concat_kv_cache));
                continue;
            }
            if let Some(tensor) = safetensors.tensor(&format!("layer_{}_conv_cache", i)).ok() {
                cache[i] = None;
            }
        }

        Ok(cache)
    }
}

fn get_file_meta(dir_entry: std::io::Result<fs::DirEntry>, model_name: &str) -> anyhow::Result<Option<CacheFileMetadata>> {
    let dir_entry = dir_entry?;
    let metadata = dir_entry.metadata()?;
    let mut cache_meta_file = None;
    let entry = dir_entry;
    if metadata.is_file() && entry.path().extension().map_or(false, |ext| ext == "safetensors") {
        cache_meta_file = Some(CacheFileMetadata {
            cache_file_path: entry.path(),
            modified: metadata.modified()?,
            file_size: metadata.len(),
        });
    }
    Ok(cache_meta_file)
}

fn get_file_meta_with_cache_meta(dir_entry: std::io::Result<fs::DirEntry>, model_name: &str) -> anyhow::Result<(Option<CacheMetadata>, Option<CacheFileMetadata>)> {
    let dir_entry = dir_entry?;
    let metadata = dir_entry.metadata()?;
    let mut cache_meta = None;
    let mut cache_meta_file = None;
    let entry = dir_entry;
    if metadata.is_file() && entry.path().extension().map_or(false, |ext| ext == "safetensors") {
        cache_meta_file = Some(CacheFileMetadata {
            cache_file_path: entry.path(),
            modified: metadata.modified()?,
            file_size: metadata.len(),
        });
        let file = File::open(entry.path())?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let (_, safetensors_meta) = SafeTensors::read_metadata(&mmap)?;
        if let Some(metadata_model_name) = safetensors_meta.metadata().as_ref().and_then(|meta| meta.get("model_name")) {
            if model_name == metadata_model_name.as_str() {
                let safetensors = SafeTensors::deserialize(&mmap)?;
                let token_ids = read_token_ids(&mmap, safetensors_meta)?;
                cache_meta = Some(CacheMetadata {
                    model_name: model_name.to_string(),
                    token_ids,
                    cache_file_path: entry.path(),
                })
            }
        }
    }
    Ok((cache_meta, cache_meta_file))
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