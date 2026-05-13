use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Linear, Module};
use candle_nn::ops::{softmax, silu, sigmoid};
use candle_nn::kv_cache::ConcatKvCache;
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::quantized_var_builder::VarBuilder;

#[cfg(feature = "cuda")]
use candle_flash_attn::flash_attn;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use nvtx::{range, range_push, range_pop};
use std::sync::Arc;
use once_cell::sync::OnceCell;

use crate::model::{Model, MixedCache};
use crate::model::utility::{get_tensor, find_norm_prefix, repeat_kv, nonzero_1d, RotaryEmbedding};
use crate::config::ModelConfig;

pub struct Deepseek2Model {
    embed_layer: Embedding,
    layers: Vec<Deepseek2Layer>,
    embed_norm: RmsNorm,
    lm_head: Linear,
    compute_dtype: DType,
}

pub struct Deepseek2Layer {
    attn_norm: RmsNorm,
    attention: Deepseek2Attn,
    ffn_norm: RmsNorm,
    ffn_mixer: Deepseek2FfnMixer,
}

impl Deepseek2Layer {
    fn forward(
        &self,
        input: &Tensor,
        cache: &mut Option<MixedCache>,
        pos: usize,
        compute_dtype: DType,
        use_flash: bool,
    ) -> anyhow::Result<Tensor> {
        // 1. Attention with pre-norm and residual
        let x = self.attn_norm.forward(&input.to_dtype(DType::F32)?)?;
        
        let attn = self.attention.forward(&x, cache, pos, compute_dtype, use_flash)?;

        let attn_output = (attn + input)?;

        // 2. FFNMixer with pre-norm and residual
        let x = self.ffn_norm.forward(&attn_output.to_dtype(DType::F32)?)?;

        let ffn_output = match &self.ffn_mixer {
            Deepseek2FfnMixer::Dense(dense_mixer) => dense_mixer.forward(&x)?,
            Deepseek2FfnMixer::Moe(moe_mixer) => moe_mixer.forward(&x)?,
        };

        let output = (ffn_output + attn_output)?;

        Ok(output)
    }
}

pub struct Deepseek2Attn {
    q_compression: Linear,
    q_norm: RmsNorm,
    q_decompression: Linear,
    kv_compression: Linear,
    kv_norm: RmsNorm,
    k_decompression: Linear,
    v_decompression: Linear,
    o_proj: Linear,
    rope: Arc<RotaryEmbedding>,
    rope_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    max_seq_len: usize,
    mask_indices: OnceCell<Tensor>,
    qk_head_dim: usize,
    qk_nope_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
}

impl Deepseek2Attn {
    fn forward(
        &self,
        x: &Tensor,
        cache: &mut Option<MixedCache>,
        pos: usize,
        compute_dtype: DType,
        use_flash: bool,
    ) -> anyhow::Result<Tensor> {
        let x = x.to_dtype(compute_dtype)?;
        let (b, s, _) = x.dims3()?;

        let q = self.q_compression.forward(&x)?;
        let q = self.q_norm.forward(&q.to_dtype(DType::F32)?)?; // q_a_layernorm
        let q = self.q_decompression.forward(&q.to_dtype(compute_dtype)?)?; // q_b_proj
        let q = q.reshape((b, s, self.n_heads, self.qk_head_dim))?.transpose(1, 2)?;

        // Split Query into NoPE and RoPE 
        let (q_pass, q_rot) = (
            q.narrow(candle_core::D::Minus1, 0, self.qk_nope_head_dim)?, 
            q.narrow(candle_core::D::Minus1, self.qk_nope_head_dim, self.rope_dim)?
        );
        let q_rot_applied = self.rope.forward_interleaved(&q_rot, pos)?;
        let q_states = Tensor::cat(&[q_pass, q_rot_applied], candle_core::D::Minus1)?;

        let compressed_kv = self.kv_compression.forward(&x)?;
        let (kv_latent, k_rot) = (
            compressed_kv.narrow(candle_core::D::Minus1, 0, self.kv_lora_rank)?, 
            compressed_kv.narrow(candle_core::D::Minus1, self.kv_lora_rank, self.rope_dim)?
        );         

        let kv_latent_norm = self.kv_norm.forward(&kv_latent.to_dtype(DType::F32)?)?; 

        // Project the shared latent to get unique NOPE parts for K and V
        let k_nope = self.k_decompression.forward(&kv_latent_norm.to_dtype(compute_dtype)?)?; // [b, s, 20 * 192]
        let value_states = self.v_decompression.forward(&kv_latent_norm.to_dtype(compute_dtype)?)?; // [b, s, 20 * 256]

        // Reshape and Transpose to (Batch, Heads, Seq, Dim)
        let k_nope = k_nope.reshape((b, s, self.n_kv_heads, self.qk_nope_head_dim))?.transpose(1, 2)?;
        let value_states = value_states.reshape((b, s, self.n_kv_heads, self.v_head_dim))?.transpose(1, 2)?;

        // Assemble the full Key (Key = K_nope + K_rope)
        let k_rot = k_rot.reshape((b, 1, s, self.rope_dim))?.transpose(1, 2)?; 
        let k_rot_applied = self.rope.forward_interleaved(&k_rot, pos)?;
        let key_states = Tensor::cat(&[
            k_nope, 
            k_rot_applied.broadcast_as((b, self.n_kv_heads, s, self.rope_dim))?
        ], candle_core::D::Minus1)?;

        let (k_states, v_states) = self.update_cache(key_states, value_states, cache)?;

        range_push!("Compute attn");
        let attn = self.compute_attention(q_states, k_states, v_states, compute_dtype, use_flash)?;
        range_pop!();

        // Reshape and project output: [B, S, H, D] -> [B, S, hidden_size]
        let attn = attn.reshape((b, s, self.v_head_dim * self.n_heads))?;
        Ok(self.o_proj.forward(&attn)?)
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

    fn compute_attention(
        &self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        compute_dtype: DType,
        #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
        use_flash: bool,
    ) -> anyhow::Result<Tensor> {
        let scale = 1.0 / (self.qk_head_dim as f64).sqrt();
        
        // Try flash attention on CUDA if available
        #[cfg(feature = "cuda")]
        {
            if use_flash && q.device().is_cuda() {
                if self.qk_head_dim > self.v_head_dim {
                    let mut v_shape = v.dims().to_vec();
                    let pad_len = self.qk_head_dim - self.v_head_dim;
                    let v_shape_len = v_shape.len();
                    v_shape[v_shape_len - 1] = pad_len;
                    let pad = Tensor::zeros(v_shape, v.dtype(), &v.device())?;
                    let v = Tensor::cat(&[v, pad], candle_core::D::Minus1)?;

                    let attn =  self.compute_flash_attention(q, k, v, scale, compute_dtype)?;
                    let attn = attn.narrow(candle_core::D::Minus1, 0, self.v_head_dim)?;
                    return Ok(attn);
                } else {
                    return self.compute_flash_attention(q, k, v, scale, compute_dtype);
                }
            }
        }

        self.compute_eager_attention(q, k, v, scale, compute_dtype)
    }   

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

    fn compute_eager_attention(
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

pub enum Deepseek2FfnMixer {
    Dense(DenseMixer),
    Moe(MoeMixer)
}


pub struct DenseMixer {
    up_proj: Linear,
    gate_proj: Linear,
    down_proj: Linear,
    compute_dtype: DType,
}

impl DenseMixer {
    fn forward(&self, input: &Tensor) -> anyhow::Result<Tensor> {
        let input = input.to_dtype(self.compute_dtype)?;

        // 1. Up and gate projections
        let up = self.up_proj.forward(&input)?;
        let gate = self.gate_proj.forward(&input)?;

        // 2. SwiGLU activation: silu(gate) * up (computed in F32 for stability)
        let gate_activated = silu(&gate.to_dtype(DType::F32)?)?;
        let gated = (gate_activated.to_dtype(self.compute_dtype)? * up)?;

        // 3. Down projection
        Ok(self.down_proj.forward(&gated)?)       
    }
}

pub struct MoeMixer {
    gate_inp: Linear,
    e_score_correction: Tensor,
    shared_mlp: DenseMixer,
    expert_gate_up: Tensor,        // blk.N.ffn_gate_exps + ffn_up_exps (cat these)
    expert_down: Tensor,           // blk.N.ffn_down_exps
    n_experts: usize,              // 64
    n_expert_used: usize,          // 4 (top_k)
    n_group: usize,                // 1
    topk_group: usize,             // 1
    routed_scaling_factor: f64,    // 1.8
    norm_top_k: bool,  
    compute_dtype: DType, 
}

impl MoeMixer {
    fn forward(&self, input: &Tensor) -> anyhow::Result<Tensor> {
        let (b, s, h) = input.dims3()?;

        // Shared Expert (Standard MLP)
        let shared_out = self.shared_mlp.forward(input)?;

        // Gate routing
        let router_logits = self.gate_inp.forward(input)?;

        let (topk_indices, topk_weights) = self.route_tokens_to_experts(&router_logits)?;

        let x_flat = input.reshape((b * s, h))?.to_dtype(self.compute_dtype)?;

        let final_routed = self.experts_forward(&x_flat, topk_indices, topk_weights)?;

        Ok((shared_out + final_routed.reshape((b, s, h)))?)

    }

    fn route_tokens_to_experts(&self, router_logits: &Tensor) -> anyhow::Result<(Tensor, Tensor)> {
        // Sigmoid and apply bias
        let router_logits = sigmoid(router_logits)?;
        let router_logits_for_choice = router_logits.broadcast_add(&self.e_score_correction)?;

        // Group Scores Calculation
        let (tokens, _) = router_logits_for_choice.dims2()?;
        let experts_per_group = self.n_experts / self.n_group;
        let reshaped = router_logits_for_choice.reshape((tokens, self.n_group, experts_per_group))?;
    
        // Sort to get top 2 values
        let (sorted_group_experts, _) = reshaped.sort_last_dim(false)?; // Descending
        let group_scores = sorted_group_experts
            .narrow(candle_core::D::Minus1, 0, 2)? // Take top 2
            .sum(candle_core::D::Minus1)?;          // Sum them

        // Group Index Selection
        let group_idx = group_scores
            .arg_sort_last_dim(false)? // Descending sort
            .narrow(candle_core::D::Minus1, 0, self.topk_group)?;

        // Create Group Mask using scatter_add
        let group_mask = Tensor::zeros_like(&group_scores)?
            .scatter_add(
                &group_idx, 
                &Tensor::ones_like(&group_idx.to_dtype(group_scores.dtype())?)?, 
                candle_core::D::Minus1
            )?;

        // Expand Mask to Expert Score Mask
        let score_mask = group_mask
            .unsqueeze(candle_core::D::Minus1)?
            .expand((tokens, self.n_group, experts_per_group))?
            .reshape((tokens, self.n_experts))?;

        // Masked Fill
        let scores_for_choice = score_mask.where_cond(
            &router_logits_for_choice, 
            &Tensor::full(f32::NEG_INFINITY, router_logits_for_choice.shape(), router_logits_for_choice.device())?
        )?;

        // Get TopK Indices for Experts
        let topk_indices = scores_for_choice
            .arg_sort_last_dim(false)?
            .narrow(candle_core::D::Minus1, 0, self.n_expert_used)?;

        // Gather Weights
        let mut topk_weights = router_logits.gather(&topk_indices, candle_core::D::Minus1)?;

        // Normalize and Scale
        if self.norm_top_k {
            let denominator = (topk_weights.sum_keepdim(candle_core::D::Minus1)? + 1e-20)?;
            topk_weights = topk_weights.broadcast_div(&denominator)?;
        }
        topk_weights = (topk_weights * self.routed_scaling_factor)?;

        Ok((topk_indices, topk_weights))

    }

    fn experts_forward(&self, input: &Tensor, topk_indices: Tensor, topk_weights: Tensor) -> anyhow::Result<Tensor> {
        let mut final_hidden_states = Tensor::zeros_like(&input)?;

        // 1. One-hot and Masking
        let expert_mask = Tensor::zeros((topk_indices.dims()[0], topk_indices.dims()[1], self.n_experts), self.compute_dtype, input.device())?
            .scatter_add(&topk_indices, &Tensor::ones_like(&topk_indices)?, 2)?;
        
        // 2. Identify active experts
        let mask_sum = expert_mask.sum_keepdim(0)?.sum_keepdim(1)?.flatten_all()?;
        let expert_indices = mask_sum.to_vec1::<f32>()?;

        for (expert_idx, &hit_count) in expert_indices.iter().enumerate() {
            if hit_count <= 0.0 || expert_idx >= self.n_experts {
                continue;
            }

            // 3. Find tokens for this expert
            let current_expert_mask = expert_mask.narrow(2, expert_idx, 1)?.squeeze(2)?;
            let flat_indices = nonzero_1d(&current_expert_mask)?;

            if flat_indices.dims1()? == 0 {
                continue;
            }

            // 4. Expert Computation
            let current_state = input.index_select(&flat_indices, 0)?;
            
            // Gate & Up projection - extract expert weights [hidden, 2*intermediate]
            let gate_up_w = self.expert_gate_up.narrow(2, expert_idx, 1)?.squeeze(2)?; 
            let gate_up = current_state.matmul(&gate_up_w.t()?)?;

            let [gate, up] = gate_up.chunk(2, candle_core::D::Minus1)?.try_into().unwrap();
            let current_hidden = (silu(&gate)? * up)?;

            // Down proj - extract expert weights [hidden, hidden]
            let down_w = self.expert_down.narrow(2, expert_idx, 1)?.squeeze(2)?;
            let current_hidden = current_hidden.matmul(&down_w.t()?)?;
            
            // 5. Weighting and Accumulation
            let current_weights = topk_weights
                .index_select(&flat_indices, 0)?
                .reshape((flat_indices.dims1()?, 1))?;
            let weighted_hidden = current_hidden.broadcast_mul(&current_weights)?;
            
            final_hidden_states = final_hidden_states.index_add(&flat_indices.into(), &weighted_hidden, 0)?;
        }

        Ok(final_hidden_states)

    }
}

impl Model for Deepseek2Model {
    fn init_cache(&self) -> anyhow::Result<Vec<Option<crate::model::MixedCache>>> {
        let caches = self
            .layers
            .iter()
            .map(|layer| Some(MixedCache::KvCache(ConcatKvCache::new(2))))
            .collect();
        Ok(caches)
    }

    fn forward(
        &self,
        input: &Tensor,
        cache: &mut Vec<Option<MixedCache>>,
        pos: usize,
        use_flash: bool,
    ) -> anyhow::Result<Tensor>
    {
        // 1. Token embeddings
        range_push!("Embed step");
        let mut x = self.embed_layer.forward(input)?;
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
        let x = self.embed_norm.forward(&x.to_dtype(DType::F32)?)?;
        let logits = self.lm_head.forward(&x.to_dtype(self.compute_dtype)?)?;

        Ok(logits)
    }
}

impl Deepseek2Model {
    pub fn load(
        config: ModelConfig,
        var_builder: VarBuilder,
        compute_dtype: DType,
        requested_max_seq_len: usize,
    ) -> anyhow::Result<Self> {
        let _range = range!("Deepseek2Model loading");

        let device = var_builder.device();
        let hidden_size = config.hidden_size;
        let n_heads = config.n_heads;
        let rope_dim = config.n_rope_dims.unwrap_or(hidden_size / n_heads);

        // Use the smaller of requested and configured max sequence length
        let effective_max_seq_len = config.max_seq_len.min(requested_max_seq_len);

        // Load embeddings (shared between input and output if no separate output.weight)
        let embed_tensor =
            get_tensor("token_embd.weight", var_builder.clone(), compute_dtype)?;

        let lm_head_tensor = if var_builder.contains_key("output.weight") {
            get_tensor("output.weight", var_builder.clone(), compute_dtype)?
        } else {
            embed_tensor.clone()
        };

        let embed_layer = Embedding::new(embed_tensor, hidden_size);
        let lm_head = Linear::new(lm_head_tensor, None);

        // Load embedding normalization
        let embed_norm_prefix = find_norm_prefix(var_builder.clone());
        let embed_norm =
            RmsNorm::new(hidden_size, config.rms_epsilon as f64, var_builder.pp(embed_norm_prefix))?;

        // Initialize rotary position embeddings
        let rope = Arc::new(RotaryEmbedding::new(
            config.rope_theta,
            rope_dim,
            effective_max_seq_len,
            true,
            device,
        )?);

        // Load all transformer layers in parallel
        let rope_clone = Arc::clone(&rope);
        let layers = (0..config.n_layers)
            .into_par_iter()
            .map(|i| {
                Self::load_layer(
                    var_builder.clone(),
                    i,
                    rope_clone.clone(),
                    rope_dim,
                    effective_max_seq_len,
                    &config,
                    compute_dtype,
                )
            })
            .collect::<anyhow::Result<Vec<Deepseek2Layer>>>()?;

        Ok(Self {
            embed_layer,
            layers,
            embed_norm,
            lm_head,
            compute_dtype,
        })
    }

    fn load_layer(
        var_builder: VarBuilder,
        block_index: usize,
        rope: Arc<RotaryEmbedding>,
        rope_dim: usize,
        effective_max_seq_len: usize,
        config: &ModelConfig,
        compute_dtype: DType,
    ) -> anyhow::Result<Deepseek2Layer> {
        let vb_l = var_builder.pp(format!("blk.{}", block_index));

        let n_leading_dense_block = config.n_leading_dense_block.unwrap_or(0);
        let rms_epsilon = config.rms_epsilon as f64;
        let hidden_size = config.hidden_size; 

        let attn_norm = RmsNorm::new(hidden_size, rms_epsilon, vb_l.pp("attn_norm"))?;

        let attention = Self::load_attention(
            &vb_l, 
            rope, 
            rope_dim,
            block_index,
            &config,
            effective_max_seq_len,
            rms_epsilon, 
            compute_dtype,
        )?;

        let ffn_norm = RmsNorm::new(hidden_size, rms_epsilon, vb_l.pp("ffn_norm"))?;

        let ffn_mixer = Self::load_ffn_mixer(
            &vb_l, 
            block_index, 
            n_leading_dense_block, 
            &config, 
            compute_dtype,
        )?;

        Ok(Deepseek2Layer {
            attn_norm,
            attention,
            ffn_norm,
            ffn_mixer,
        })

    }

    fn load_attention(
        var_builder: &VarBuilder,
        rope: Arc<RotaryEmbedding>,
        rope_dim: usize,
        block_index: usize,
        config: &ModelConfig,
        effective_max_seq_len: usize,
        rms_epsilon: f64,
        compute_dtype: DType,
    ) -> anyhow::Result<Deepseek2Attn> {
        let q_lora_rank = config.q_lora_rank.ok_or_else(||
            anyhow::anyhow!("Missing q_lora_rank, which is required for DeepSeek2 model"))?;
        let kv_lora_rank = config.kv_lora_rank.ok_or_else(||
            anyhow::anyhow!("Missing kv_lora_rank, which is required for DeepSeek2 model"))?;
        let attn_key_length_mla = config.attn_key_length_mla.ok_or_else(||
            anyhow::anyhow!("Missing attn_key_length_mla, which is required for DeepSeek2 model"))?;
        let attn_value_length_mla = config.attn_value_length_mla.ok_or_else(||
            anyhow::anyhow!("Missing attn_value_length_mla, which is required for DeepSeek2 model"))?;
        let n_kv_heads = config.n_kv_heads[block_index];

        Ok(Deepseek2Attn {
            q_compression: Linear::new(
                get_tensor("attn_q_a.weight", var_builder.clone(), compute_dtype)?, 
                None
            ),
            q_norm: RmsNorm::new(q_lora_rank, rms_epsilon, var_builder.pp("attn_q_a_norm"))?,
            q_decompression: Linear::new(
                get_tensor("attn_q_b.weight", var_builder.clone(), compute_dtype)?, 
                None
            ),
            kv_compression: Linear::new(
                get_tensor("attn_kv_a_mqa.weight", var_builder.clone(), compute_dtype)?, 
                None
            ),
            kv_norm: RmsNorm::new(kv_lora_rank, rms_epsilon, var_builder.pp("attn_q_a_norm"))?,
            k_decompression: Linear::new(
                get_tensor("attn_k_b.weight", var_builder.clone(), compute_dtype)?, 
                None
            ),
            v_decompression: Linear::new(
                get_tensor("attn_v_b.weight", var_builder.clone(), compute_dtype)?, 
                None
            ),
            o_proj: Linear::new(
                get_tensor("attn_output.weight", var_builder.clone(), compute_dtype)?, 
                None
            ),
            rope,
            rope_dim,
            n_heads: config.n_heads,
            n_kv_heads,
            max_seq_len: effective_max_seq_len,
            mask_indices: OnceCell::new(),
            qk_head_dim: attn_key_length_mla,
            qk_nope_head_dim: attn_key_length_mla - rope_dim,
            v_head_dim: attn_value_length_mla,
            kv_lora_rank: kv_lora_rank,
        })
    }

    fn load_ffn_mixer(
        var_builder: &VarBuilder,
        block_index: usize,
        n_leading_dense_block: usize,
        config: &ModelConfig,
        compute_dtype: DType,
    ) -> anyhow::Result<Deepseek2FfnMixer> {
        let ffn_mixer = if block_index < n_leading_dense_block {
            Deepseek2FfnMixer::Dense(Self::load_dense_mixer(var_builder, compute_dtype)?)
        } else {
            Deepseek2FfnMixer::Moe(Self::load_moe_mixer(var_builder, config, compute_dtype)?)
        };

        Ok(ffn_mixer)
    }

    fn load_dense_mixer(var_builder: &VarBuilder, compute_dtype: DType) -> anyhow::Result<DenseMixer> {
        Ok(DenseMixer {
            up_proj: Linear::new(
                get_tensor("ffn_up.weight", var_builder.clone(), compute_dtype)?,
                None,
            ),
            gate_proj: Linear::new(
                get_tensor("ffn_gate.weight", var_builder.clone(), compute_dtype)?,
                None,
            ),
            down_proj: Linear::new(
                get_tensor("ffn_down.weight", var_builder.clone(), compute_dtype)?,
                None,
            ),
            compute_dtype,
        })
    }

    fn load_moe_mixer(var_builder: &VarBuilder, config: &ModelConfig, compute_dtype: DType) -> anyhow::Result<MoeMixer> {
        let n_experts = config.n_experts.ok_or_else(||
            anyhow::anyhow!("Missing n_experts, which is required for DeepSeek2 model"))?;
        let n_expert_used = config.n_expert_used.ok_or_else(||
            anyhow::anyhow!("Missing n_expert_used, which is required for DeepSeek2 model"))?;
        let n_group = config.n_expert_group.ok_or_else(||
            anyhow::anyhow!("Missing n_expert_group, which is required for DeepSeek2 model"))?;
        let topk_group = config.n_expert_group_used.ok_or_else(||
            anyhow::anyhow!("Missing n_expert_group_used, which is required for DeepSeek2 model"))?;
        let norm_top_k = config.expert_weights_norm.unwrap_or(false);
        let routed_scaling_factor = config.expert_weights_scale.ok_or_else(||
            anyhow::anyhow!("Missing expert_weights_scale, which is required for DeepSeek2 model"))? as f64;
        
        Ok(MoeMixer {
            gate_inp: Linear::new(
                get_tensor("ffn_gate_inp.weight", var_builder.clone(), compute_dtype)?,
                None, 
            ),
            e_score_correction: get_tensor("exp_probs_b.bias", var_builder.clone(), compute_dtype)?,
            shared_mlp: DenseMixer {
                up_proj: Linear::new(
                    get_tensor("ffn_up_shexp.weight", var_builder.clone(), compute_dtype)?,
                    None,
                ),
                gate_proj: Linear::new(
                    get_tensor("ffn_gate_shexp.weight", var_builder.clone(), compute_dtype)?,
                    None,
                ),
                down_proj: Linear::new(
                    get_tensor("ffn_down_shexp.weight", var_builder.clone(), compute_dtype)?,
                    None,
                ),
                compute_dtype,
            },
            expert_gate_up: Tensor::cat(&[
                get_tensor("ffn_gate_exps.weight", var_builder.clone(), compute_dtype)?,
                get_tensor("ffn_up_exps.weight", var_builder.clone(), compute_dtype)?
            ], 
            candle_core::D::Minus2)?,  // Each is [hidden_size, intermediate, n_experts]; concat on dim 1 → [hidden_size, 2*intermediate, n_experts]
            expert_down: get_tensor("ffn_down_exps.weight", var_builder.clone(), compute_dtype)?,
            n_experts,
            n_expert_used,
            n_group,
            topk_group,
            routed_scaling_factor,
            norm_top_k,
            compute_dtype,
        })
    }

}