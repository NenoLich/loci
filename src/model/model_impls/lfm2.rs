use candle_core::quantized::{GgmlDType, QMatMul};
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "cuda")]
use candle_flash_attn::flash_attn;
use candle_nn::kv_cache::ConcatKvCache;
use candle_nn::{
    Conv1d, Conv1dConfig, Module,
    ops::{silu, softmax},
};
use nvtx::{range, range_push, range_pop};
use candle_transformers::quantized_nn::{RmsNorm, Embedding};
use candle_transformers::quantized_var_builder::VarBuilder;
use once_cell::sync::OnceCell;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;

use crate::model::utility::{repeat_kv, get_tensor, find_norm_prefix, qmatmul_forward, RotaryEmbedding};
use crate::{
    model::{Model, MixedCache},
    config::ModelConfig,
};

/// LFM2 transformer model implementation.
///
/// This model consists of token embeddings, a series of transformer layers
/// (with mixed attention/convolution mixers), and an output head.
/// It supports quantized weights and various compute dtypes (F16, BF16, F32).
pub struct Lfm2Model {
    /// Token embedding layer (token_embd.weight)
    pub embed_layer: Embedding,
    /// Embedding normalization (token_embd_norm.weight) - LFM2 specific
    pub embed_norm: RmsNorm,
    /// Transformer layers (typically 16 blocks)
    pub layers: Vec<Lfm2Layer>,
    /// Output projection layer (output.weight)
    pub lm_head: QMatMul,
    /// Data type used for computations (F16, BF16, or F32)
    pub compute_dtype: DType,
    /// kv sequence lentgh dimension
    pub cache_seq_len_dim: usize,
    /// Minimum input tokens needed for conv cache init (max conv_l_cache - 1 across layers)
    pub min_prefill_tokens: usize,
}

/// A single transformer layer in the LFM2 model.
///
/// Each layer consists of:
/// 1. An attention/convolution mixer (the "motor")
/// 2. Feed-forward network (FFN) with SwiGLU activation
/// 3. RMS normalization before attention and FFN
pub struct Lfm2Layer {
    /// Pre-attention normalization (blk.N.attn_norm.weight)
    pub attn_norm: RmsNorm,
    /// Mixed attention/convolution block (the "motor")
    pub mixer: Lfm2Mixer,
    /// Pre-FFN normalization (blk.N.ffn_norm.weight)
    pub ffn_norm: RmsNorm,
    /// Feed-forward network with gate, up, and down projections
    pub ffn: Lfm2Ffn,
}

impl Lfm2Layer {
    /// Forward pass through a single transformer layer.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch, seq_len, hidden_size]
    /// * `cache` - KV cache or convolution cache for incremental decoding
    /// * `pos` - Current position in the sequence (for RoPE)
    /// * `compute_dtype` - Data type for computations
    /// * `use_flash` - Whether to use flash attention (if available)
    ///
    /// # Returns
    /// Output tensor of shape [batch, seq_len, hidden_size]
    fn forward(
        &self,
        input: &Tensor,
        cache: &mut Option<MixedCache>,
        pos: usize,
        compute_dtype: DType,
        use_flash: bool,
    ) -> anyhow::Result<Tensor> {
        // 1. Attention/convolution mixer with pre-norm and residual
        let x = self.attn_norm.forward(&input.to_dtype(DType::F32)?)?;
        let mixed = match &self.mixer {
            Lfm2Mixer::Attention(attn_mixer) => {
                attn_mixer.forward(&x, cache, pos, compute_dtype, use_flash)?
            }
            Lfm2Mixer::ShortConv(conv_mixer) => {
                conv_mixer.forward(&x, cache, compute_dtype)?
            }
        };
        let attn_output = (mixed + input)?;

        // 2. FFN with pre-norm and residual
        let x = self.ffn_norm.forward(&attn_output.to_dtype(DType::F32)?)?;
        let ffn_output = self.ffn.forward(&x.to_dtype(compute_dtype)?)?;
        let output = (ffn_output + attn_output)?;

        Ok(output)
    }
}

/// The mixer (or "motor") of a transformer layer.
///
/// LFM2 uses a hybrid architecture where some layers use standard attention
/// and others use a short convolution. This enum allows each layer to
/// select its mixer type.
pub enum Lfm2Mixer {
    /// Multi-head attention with RoPE and QK-normalization
    Attention(AttentionMixer),
    /// Short convolution with gating mechanism
    ShortConv(ShortConvMixer),
}

/// Multi-head attention mixer with RoPE and QK-normalization.
///
/// This implements grouped-query attention (GQA) where the number of KV heads
/// may be less than the number of query heads. It also applies QK-normalization
/// which is specific to LFM2.
pub struct AttentionMixer {
    /// Query projection (blk.N.attn_q.weight)
    q_proj: QMatMul,
    /// Key projection (blk.N.attn_k.weight)
    k_proj: QMatMul,
    /// Value projection (blk.N.attn_v.weight)
    v_proj: QMatMul,
    /// Output projection (blk.N.attn_output.weight)
    o_proj: QMatMul,
    /// Query normalization (blk.N.attn_q_norm) - LFM2 specific
    q_norm: RmsNorm,
    /// Key normalization (blk.N.attn_k_norm) - LFM2 specific
    k_norm: RmsNorm,
    /// Cached attention mask indices (lazily initialized)
    mask_indices: OnceCell<Tensor>,
    /// Rotary position embeddings
    rope: Arc<RotaryEmbedding>,
    /// Dimension of each attention head
    head_dim: usize,
    /// Number of query heads
    n_heads: usize,
    /// Number of key/value heads (for grouped-query attention)
    n_kv_heads: usize,
    /// Maximum sequence length for attention mask
    max_seq_len: usize,
}

impl AttentionMixer {
    /// Forward pass through the attention mixer.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, seq_len, hidden_size]
    /// * `cache` - KV cache for incremental decoding
    /// * `pos` - Current position in the sequence (for RoPE)
    /// * `compute_dtype` - Data type for computations
    /// * `use_flash` - Whether to use flash attention (if available on CUDA)
    ///
    /// # Returns
    /// Output tensor of shape [batch, seq_len, hidden_size]
    pub fn forward(
        &self,
        x: &Tensor,
        cache: &mut Option<MixedCache>,
        pos: usize,
        compute_dtype: DType,
        use_flash: bool,
    ) -> anyhow::Result<Tensor> {
        let x = x.to_dtype(compute_dtype)?;

        // 1. Project input to Q, K, V
        let (batch_size, seq_len, _) = x.dims3()?;
        let (q, k, v) = self.project_qkv(&x)?;
        
        // 2. Apply RoPE to Q and K
        let q = self.rope.forward(&q, pos)?;
        let k = self.rope.forward(&k, pos)?;

        // 3. Apply QK-normalization (LFM2 specific: Q uses q_norm, K uses k_norm)
        let q = self.q_norm.forward(&q.to_dtype(DType::F32)?)?;
        let k = self.k_norm.forward(&k.to_dtype(DType::F32)?)?;

        // 4. Update KV cache and compute attention
        let (k, v) = self.update_cache(k, v, cache)?;

        range_push!("Compute attn");
        let y = self.compute_attention(q, k, v, compute_dtype, use_flash)?;
        range_pop!();

        // 5. Reshape and project output: [B, S, H, D] -> [B, S, hidden_size]
        let y = y.reshape((batch_size, seq_len, self.head_dim * self.n_heads))?;
        Ok(qmatmul_forward(&self.o_proj, &y)?)
    }

    /// Project input to query, key, and value tensors with head dimensions.
    fn project_qkv(&self, x: &Tensor) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        let q = qmatmul_forward(&self.q_proj, x)?;
        let k = qmatmul_forward(&self.k_proj, x)?;
        let v = qmatmul_forward(&self.v_proj, x)?;

        let (batch_size, seq_len, _) = q.dims3()?;

        let q = q
            .reshape((batch_size, seq_len, self.n_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let k = k
            .reshape((batch_size, seq_len, self.n_kv_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let v = v
            .reshape((batch_size, seq_len, self.n_kv_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;

        Ok((q, k, v))
    }

    /// Update KV cache with new keys and values.
    fn update_cache(
        &self,
        k: Tensor,
        v: Tensor,
        cache: &mut Option<MixedCache>,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        match cache {
            Some(MixedCache::KvCache(kv_cache)) => {
                kv_cache.append(&k.to_dtype(DType::F16)?, &v.to_dtype(DType::F16)?).map_err(anyhow::Error::msg)
            }
            _ => Ok((k, v)),
        }
    }

    /// Compute attention scores and apply attention weights to values.
    ///
    /// Uses either flash attention (on CUDA) or standard grouped-query attention.
    fn compute_attention(
        &self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        compute_dtype: DType,
        #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
        use_flash: bool,
    ) -> anyhow::Result<Tensor> {
        let scale = 1.0 / (self.head_dim as f64).sqrt();

        // Try flash attention on CUDA if available
        #[cfg(feature = "cuda")]
        {
            if use_flash && q.device().is_cuda() {
                return self.compute_flash_attention(q, k, v, scale, compute_dtype);
            }
        }

        // Fall back to standard grouped-query attention
        self.compute_gqa(q, k, v, scale, compute_dtype)
    }

    /// Compute attention using flash attention (CUDA only).
    #[cfg(feature = "cuda")]
    fn compute_flash_attention(
        &self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        scale: f64,
        compute_dtype: DType,
    ) -> anyhow::Result<Tensor> {
        // Flash attention expects [Batch, Seq, Heads, Head_Dim]
        let q = q.transpose(1, 2)?.to_dtype(DType::BF16)?.contiguous()?;
        let k = k.transpose(1, 2)?.to_dtype(DType::BF16)?.contiguous()?;
        let v = v.transpose(1, 2)?.to_dtype(DType::BF16)?.contiguous()?;

        // causal: true handles the masking
        let attn = flash_attn(&q, &k, &v, scale as f32, true)?;
        Ok(attn.to_dtype(compute_dtype)?)
    }

    /// Compute grouped-query attention (GQA) with causal masking.
    fn compute_gqa(
        &self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        scale: f64,
        compute_dtype: DType,
    ) -> anyhow::Result<Tensor> {
        let (_batch, _heads, seq_len, _dim) = q.dims4()?;

        // Repeat KV heads for grouped-query attention
        let n_repeat = self.n_heads / self.n_kv_heads;
        let k = repeat_kv(k, n_repeat)?.to_dtype(DType::F32)?;
        let v = repeat_kv(v, n_repeat)?.to_dtype(DType::F32)?;

        // Compute attention scores: Q @ K.T / sqrt(head_dim)
        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        // Apply causal mask for sequences longer than 1 token
        let attn = if seq_len > 1 {
            let mask = self.get_mask(seq_len, compute_dtype, q.device())?;
            let attn = attn.to_dtype(compute_dtype)?;
            attn.broadcast_add(&mask)?.to_dtype(DType::F32)?
        } else {
            attn
        };

        // Softmax and apply to values
        let attn_weights = softmax(&attn, candle_core::D::Minus1)?;
        let y_t = attn_weights.matmul(&v)?;

        // Cast back to compute dtype and transpose to [B, S, H, D]
        Ok(y_t.to_dtype(compute_dtype)?.transpose(1, 2)?)
    }

    /// Get or create the causal attention mask for the given sequence length.
    fn get_mask(
        &self,
        seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        // Lazily initialize the mask indices
        let indices = self.mask_indices.get_or_try_init(|| {
            Tensor::arange(0u32, self.max_seq_len as u32, device)
        })?;
        let idx = indices.narrow(0, 0, seq_len)?;

        // Create causal mask: positions i < j should be masked
        let i = idx.reshape((seq_len, 1))?.broadcast_as((seq_len, seq_len))?;
        let j = idx.reshape((1, seq_len))?.broadcast_as((seq_len, seq_len))?;
        let mask = i.lt(&j)?;

        // Apply mask: masked positions get -inf, others get 0
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?
            .to_dtype(dtype)?
            .broadcast_as((seq_len, seq_len))?;
        let zeros = Tensor::new(0f32, device)?
            .to_dtype(dtype)?
            .broadcast_as((seq_len, seq_len))?;

        Ok(mask.where_cond(&neg_inf, &zeros)?)
    }
}

/// Short convolution mixer with gating mechanism.
///
/// This mixer uses a depth-wise 1D convolution instead of attention.
/// It projects the input into three parts (B, C, H), applies gating
/// before and after the convolution, and uses a cache for efficient
/// incremental decoding.
pub struct ShortConvMixer {
    /// Input projection that splits into B, C, H (gate, gate, signal)
    in_proj: QMatMul,
    /// Depth-wise 1D convolution (groups = channels)
    conv: Conv1d,
    /// Output projection
    out_proj: QMatMul,
    /// Cache length for convolution state
    conv_l_cache: usize,
    /// Convolution kernel size
    kernel_size: usize,
    /// Whether to run convolution on CPU (useful for CUDA performance)
    conv_on_cpu: bool,
}

impl ShortConvMixer {
    /// Forward pass through the short convolution mixer.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch, seq_len, hidden_size]
    /// * `cache` - Convolution cache for incremental decoding
    /// * `compute_dtype` - Data type for computations
    ///
    /// # Returns
    /// Output tensor of shape [batch, seq_len, hidden_size]
    pub fn forward(
        &self,
        input: &Tensor,
        cache: &mut Option<MixedCache>,
        compute_dtype: DType,
    ) -> anyhow::Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = input.dims3()?;

        // 1. Project input to [B, S, 3 * hidden_size] and split into gates and signal
        let projected = qmatmul_forward(&self.in_proj, &input.to_dtype(compute_dtype)?)?;
        let chunks = projected.chunk(3, 2)?;
        let (b_gate, c_gate, h_signal) = match &chunks[..] {
            [b, c, h] => (b, c, h),
            _ => unreachable!("chunk(3, 2) always returns 3 tensors"),
        };
        
        // 2. First gating: y = B ⊙ h
        let gated = (b_gate * h_signal)?;

        // 3. Convolution with caching
        let conv_input = gated.transpose(1, 2)?;
        let conv_out = self.conv_forward(conv_input, cache, batch_size, seq_len, hidden_size)?;
        let conv_out = conv_out.to_dtype(compute_dtype)?.transpose(1, 2)?;

        // 4. Second gating: C ⊙ conv(z)
        let gated_output = (c_gate * conv_out)?;

        // 5. Output projection
        Ok(qmatmul_forward(&self.out_proj, &gated_output)?)
    }

    /// Perform convolution with caching support.
    ///
    /// Handles both prefill (seq_len > 1) and decode (seq_len == 1) modes.
    fn conv_forward(
        &self,
        input: Tensor,
        cache: &mut Option<MixedCache>,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> anyhow::Result<Tensor> {
        let _range = range!("Conv forward (cache and forward)");

        // Optionally move to CPU for convolution
        let original_device = input.device().clone();
        let input = if self.conv_on_cpu && original_device.is_cuda() {
            input.to_device(&Device::Cpu)?
        } else {
            input
        };

        // Prepare input with cache/padding
        let conv_input = self.prepare_conv_input(&input, cache, batch_size, seq_len, hidden_size)?;

        // Perform convolution in F32
        let conv_input = conv_input.to_dtype(DType::F32)?;
        let conv_out = self.conv.forward(&conv_input)?;

        // Move back to original device if needed
        let output = if self.conv_on_cpu && original_device.is_cuda() {
            conv_out.to_device(&original_device)?
        } else {
            conv_out
        };
        Ok(output)
    }

    /// Prepare convolution input with caching or padding.
    fn prepare_conv_input(
        &self,
        input: &Tensor,
        cache: &mut Option<MixedCache>,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> anyhow::Result<Tensor> {
        let pad_len = self.kernel_size - 1;

        if seq_len == 1 {
            // Decode mode: use cache or pad
            self.prepare_decode_input(input, cache, batch_size, hidden_size, pad_len)
        } else {
            // Prefill mode: pad left and update cache
            self.prepare_prefill_input(input, cache, batch_size, seq_len, hidden_size, pad_len)
        }
    }

    /// Prepare input for decode mode (single token).
    fn prepare_decode_input(
        &self,
        input: &Tensor,
        cache: &mut Option<MixedCache>,
        batch_size: usize,
        hidden_size: usize,
        pad_len: usize,
    ) -> anyhow::Result<Tensor> {
        if let Some(MixedCache::ConvCache(state)) = cache {
            // Clone the state tensor to use it
            let state_tensor = state.clone();
            // Concatenate cached state with new input
            let input_cached = Tensor::cat(&[&state_tensor, input], 2)?;
            let current_len = input_cached.dims()[2];

            // Update cache with the last (kernel_size - 1) tokens
            let new_cache = input_cached.narrow(
                2,
                current_len - (self.conv_l_cache - 1),
                self.conv_l_cache - 1,
            )?;
            *cache = Some(MixedCache::ConvCache(new_cache));

            // Use the last kernel_size tokens for convolution
            Ok(input_cached.narrow(2, current_len - self.kernel_size, self.kernel_size)?)
        } else {
            // No cache: pad left to match kernel size
            let pad = Tensor::zeros(
                (batch_size, hidden_size, pad_len),
                input.dtype(),
                input.device(),
            )?;
            Ok(Tensor::cat(&[pad, input.clone()], 2)?)
        }
    }

    /// Prepare input for prefill mode (multiple tokens).
    fn prepare_prefill_input(
        &self,
        input: &Tensor,
        cache: &mut Option<MixedCache>,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
        pad_len: usize,
    ) -> anyhow::Result<Tensor> {
        // Pad left to maintain causal convolution
        let pad = Tensor::zeros(
            (batch_size, hidden_size, pad_len),
            input.dtype(),
            input.device(),
        )?;

        // Update cache with the last pad_len tokens for future decode steps
        if seq_len >= pad_len {
            let new_cache = input.narrow(2, seq_len - (self.conv_l_cache - 1), self.conv_l_cache - 1)?;
            *cache = Some(MixedCache::ConvCache(new_cache));
        }

        Ok(Tensor::cat(&[pad, input.clone()], 2)?)
    }
}


/// Feed-forward network with SwiGLU activation.
///
/// This implements the standard FFN used in LFM2:
/// FFN(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
///
/// The activation is computed in F32 for numerical stability,
/// then cast back to the compute dtype.
pub struct Lfm2Ffn {
    /// Up projection (blk.N.ffn_up)
    pub up_proj: QMatMul,
    /// Gate projection with SwiGLU activation (blk.N.ffn_gate)
    pub gate_proj: QMatMul,
    /// Down projection (blk.N.ffn_down)
    pub down_proj: QMatMul,
    /// Data type used for computations
    pub compute_dtype: DType,
}

impl Lfm2Ffn {
    /// Forward pass through the FFN with SwiGLU activation.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch, seq_len, hidden_size]
    ///
    /// # Returns
    /// Output tensor of shape [batch, seq_len, hidden_size]
    pub fn forward(&self, input: &Tensor) -> anyhow::Result<Tensor> {
        let input = input.to_dtype(self.compute_dtype)?;

        // 1. Up and gate projections
        let up = qmatmul_forward(&self.up_proj, &input)?;
        let gate = qmatmul_forward(&self.gate_proj, &input)?;

        // 2. SwiGLU activation: silu(gate) * up (computed in F32 for stability)
        let gate_activated = silu(&gate.to_dtype(DType::F32)?)?;
        let gated = (gate_activated.to_dtype(self.compute_dtype)? * up)?;

        // 3. Down projection
        Ok(qmatmul_forward(&self.down_proj, &gated)?)
    }
}

impl Model for Lfm2Model {
    /// Forward pass through the entire LFM2 model.
    ///
    /// # Arguments
    /// * `input` - Token IDs of shape [batch, seq_len]
    /// * `cache` - Vector of caches (one per layer) for incremental decoding
    /// * `pos` - Current position in the sequence (for RoPE)
    /// * `use_flash` - Whether to use flash attention (if available)
    ///
    /// # Returns
    /// Logits of shape [batch, seq_len, vocab_size]
    fn forward(
        &self,
        input: &Tensor,
        cache: &mut Vec<Option<MixedCache>>,
        pos: usize,
        use_flash: bool,
    ) -> anyhow::Result<Tensor> {
        // 1. Token embeddings
        range_push!("Embed step");
        let mut x = self.embed_layer.forward(input)?.to_dtype(self.compute_dtype)?;
        range_pop!();
        // 2. Transformer layers
        range_push!("Layer step");
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(
                &x,
                cache.get_mut(i).unwrap(),
                pos,
                self.compute_dtype,
                use_flash,
            )?;
        }
        range_pop!();

        // 3. Final normalization and output projection
        let final_norm = self.embed_norm.forward(&x.to_dtype(DType::F32)?)?;
        let logits = qmatmul_forward(&self.lm_head, &final_norm.to_dtype(self.compute_dtype)?)?;

        Ok(logits)
    }

    /// Initialize caches for all layers.
    ///
    /// Attention layers get a KV cache, while convolution layers start with None
    /// (cache is created on the first forward pass).
    fn init_cache(&self) -> anyhow::Result<Vec<Option<MixedCache>>> {
        let caches = self
            .layers
            .iter()
            .map(|layer| match &layer.mixer {
                Lfm2Mixer::Attention(_) => {
                    Some(MixedCache::KvCache(ConcatKvCache::new(2)))
                }
                Lfm2Mixer::ShortConv(_) => None,
            })
            .collect();
        Ok(caches)
    }

    fn cache_seq_len_dim(&self) -> usize {
        self.cache_seq_len_dim
    }

    fn min_prefill_tokens(&self) -> usize {
        self.min_prefill_tokens
    }
}

impl Lfm2Model {
    /// Load and initialize the LFM2 model from GGUF weights.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `var_builder` - Variable builder for loading GGUF tensors
    /// * `compute_dtype` - Data type for computations (F16, BF16, or F32)
    /// * `requested_max_seq_len` - Requested maximum sequence length (will be capped by config)
    /// * `conv_on_cpu` - Whether to run convolutions on CPU
    ///
    /// # Returns
    /// Initialized LFM2 model
    pub fn load(
        config: ModelConfig,
        var_builder: VarBuilder,
        compute_dtype: DType,
        requested_max_seq_len: usize,
        conv_on_cpu: bool,
    ) -> anyhow::Result<Self> {
        let _range = range!("Lfm2Model loading");

        let rms_epsilon = config.rms_epsilon as f64;
        let hidden_size = config.hidden_size;
        let n_layers = config.n_layers;
        let n_heads = config.n_heads;
        let vocab_size = config.vocab_size;
        let head_dim = hidden_size / n_heads;
        let device = var_builder.device();

        // Use the smaller of requested and configured max sequence length
        let effective_max_seq_len = config.max_seq_len.min(requested_max_seq_len);

        // Load embeddings (shared between input and output if no separate output.weight)
        let lm_head_tensor = if var_builder.contains_key("output.weight") {
            var_builder.get_no_shape("output.weight")?
        } else {
            var_builder.get_no_shape("token_embd.weight")?
        };

        let embed_layer = Embedding::new(vocab_size, hidden_size, var_builder.pp("token_embd"))?;
        let lm_head = QMatMul::from_arc(lm_head_tensor)?;

        // Load embedding normalization
        let embed_norm_prefix = find_norm_prefix(var_builder.clone());
        let embed_norm =
            RmsNorm::new(hidden_size, rms_epsilon, var_builder.pp(embed_norm_prefix))?;

        // Initialize rotary position embeddings
        let rope = Arc::new(RotaryEmbedding::new(
            config.rope_theta,
            head_dim,
            effective_max_seq_len,
            false,
            device,
        )?);

        // Load all transformer layers in parallel
        let rope_clone = Arc::clone(&rope);
        let layers = (0..n_layers)
            .into_par_iter()
            .map(|i| {
                Self::load_layer(
                    var_builder.clone(),
                    i,
                    rope_clone.clone(),
                    &config,
                    effective_max_seq_len,
                    compute_dtype,
                    conv_on_cpu,
                )
            })
            .collect::<anyhow::Result<Vec<Lfm2Layer>>>()?;

        // Find minimum prefill tokens for conv layers (each conv needs kernel_size-1 context)
        let min_prefill_tokens = config.conv_l_cache.map(|c| c.saturating_sub(1)).unwrap_or(1);

        Ok(Self {
            embed_layer,
            embed_norm,
            layers,
            lm_head,
            compute_dtype,
            cache_seq_len_dim: config.cache_seq_len_dim,
            min_prefill_tokens,
        })
    }

    /// Load a single transformer layer.
    ///
    /// Determines whether the layer uses attention or convolution based on n_kv_heads.
    fn load_layer(
        var_builder: VarBuilder,
        block_index: usize,
        rope: Arc<RotaryEmbedding>,
        config: &ModelConfig,
        effective_max_seq_len: usize,
        compute_dtype: DType,
        conv_on_cpu: bool,
    ) -> anyhow::Result<Lfm2Layer> {
        let layer_prefix = format!("blk.{}", block_index);
        let var_builder = var_builder.pp(layer_prefix);
        let conv_device = if conv_on_cpu {
            Device::Cpu
        } else {
            var_builder.device().clone()
        };

        let rms_epsilon = config.rms_epsilon as f64;
        let n_kv_heads = config.n_kv_heads[block_index];
        let head_dim = config.hidden_size / config.n_heads;

        // 1. Attention normalization
        let attn_norm =
            RmsNorm::new(config.hidden_size, rms_epsilon, var_builder.pp("attn_norm"))?;

        // 2. Load mixer (attention or convolution)
        let mixer = if n_kv_heads > 0 {
            Lfm2Mixer::Attention(Self::load_attention_mixer(
                &var_builder,
                rope,
                head_dim,
                config.n_heads,
                n_kv_heads,
                effective_max_seq_len,
                rms_epsilon,
                compute_dtype,
            )?)
        } else {
            Lfm2Mixer::ShortConv(Self::load_conv_mixer(
                &var_builder,
                conv_device,
                config.conv_l_cache.unwrap_or(3),
                conv_on_cpu,
                compute_dtype,
            )?)
        };

        // 3. FFN normalization
        let ffn_norm =
            RmsNorm::new(config.hidden_size, rms_epsilon, var_builder.pp("ffn_norm"))?;

        // 4. Load FFN
        let ffn = Self::load_ffn(&var_builder, compute_dtype)?;

        Ok(Lfm2Layer {
            attn_norm,
            mixer,
            ffn_norm,
            ffn,
        })
    }

    /// Load an attention mixer for a layer.
    fn load_attention_mixer(
        var_builder: &VarBuilder,
        rope: Arc<RotaryEmbedding>,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        effective_max_seq_len: usize,
        rms_epsilon: f64,
        compute_dtype: DType,
    ) -> anyhow::Result<AttentionMixer> {
        Ok(AttentionMixer {
            q_proj: QMatMul::from_arc(
                var_builder.get_no_shape("attn_q.weight")?,
            )?,
            k_proj: QMatMul::from_arc(
                var_builder.get_no_shape("attn_k.weight")?,
            )?,
            v_proj: QMatMul::from_arc(
                var_builder.get_no_shape("attn_v.weight")?,
            )?,
            o_proj: QMatMul::from_arc(
                var_builder.get_no_shape("attn_output.weight")?,
            )?,
            q_norm: RmsNorm::new(head_dim, rms_epsilon, var_builder.pp("attn_q_norm"))?,
            k_norm: RmsNorm::new(head_dim, rms_epsilon, var_builder.pp("attn_k_norm"))?,
            rope,
            mask_indices: OnceCell::new(),
            head_dim,
            n_heads,
            n_kv_heads,
            max_seq_len: effective_max_seq_len,
        })
    }

    /// Load a convolution mixer for a layer.
    fn load_conv_mixer(
        var_builder: &VarBuilder,
        conv_device: Device,
        conv_l_cache: usize,
        conv_on_cpu: bool,
        compute_dtype: DType,
    ) -> anyhow::Result<ShortConvMixer> {
        // Load and prepare convolution weights (depth-wise, so groups = channels)
        let raw_conv_weights = var_builder.get_no_shape("shortconv.conv.weight")?;
        let (channels, kernel_size) = raw_conv_weights.shape().dims2()?;

        let conv_cfg = Conv1dConfig {
            groups: channels,
            padding: 0,
            ..Default::default()
        };

        let conv_weights = raw_conv_weights
            .dequantize(&conv_device)?
            .unsqueeze(1)?;

        Ok(ShortConvMixer {
            in_proj: QMatMul::from_arc(
                var_builder.get_no_shape("shortconv.in_proj.weight")?,
            )?,
            conv: Conv1d::new(conv_weights, None, conv_cfg),
            out_proj: QMatMul::from_arc(
                var_builder.get_no_shape("shortconv.out_proj.weight")?,
            )?,
            conv_l_cache,
            kernel_size,
            conv_on_cpu,
        })
    }

    /// Load FFN (feed-forward network) for a layer.
    fn load_ffn(var_builder: &VarBuilder, compute_dtype: DType) -> anyhow::Result<Lfm2Ffn> {
        Ok(Lfm2Ffn {
            up_proj: QMatMul::from_arc(
                var_builder.get_no_shape("ffn_up.weight")?,
            )?,
            gate_proj: QMatMul::from_arc(
                var_builder.get_no_shape("ffn_gate.weight")?,
            )?,
            down_proj: QMatMul::from_arc(
                var_builder.get_no_shape("ffn_down.weight")?,
            )?,
            compute_dtype,
        })
    }
}


