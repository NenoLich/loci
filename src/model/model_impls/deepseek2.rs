use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{DType, Tensor};
use candle_nn::Module;
use candle_nn::kv_cache::ConcatKvCache;
use candle_nn::ops::{sigmoid, silu, softmax};
use candle_transformers::quantized_nn::{Embedding, RmsNorm};
use candle_transformers::quantized_var_builder::VarBuilder;

#[cfg(feature = "cuda")]
use candle_flash_attn::flash_attn;
use once_cell::sync::OnceCell;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;
use tracing::{debug, trace_span};

use crate::config::ModelConfig;
use crate::model::utility::{
    RotaryEmbedding, find_norm_prefix, get_mask, get_tensor, qmatmul_forward, repeat_kv,
    update_cache,
};
use crate::model::{MixedCache, Model};
use crate::profiling;

pub struct Deepseek2Model {
    embed_layer: Embedding,
    layers: Vec<Deepseek2Layer>,
    embed_norm: RmsNorm,
    lm_head: QMatMul,
    compute_dtype: DType,
    cache_seq_len_dim: usize,
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

        let input = input.to_dtype(DType::F32)?;
        let x = self.attn_norm.forward(&input)?;

        let attn = self
            .attention
            .forward(&x, cache, pos, compute_dtype, use_flash)?;

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
    q_compression: QMatMul,
    q_norm: RmsNorm,
    q_decompression: QMatMul,
    kv_compression: QMatMul,
    kv_norm: RmsNorm,
    k_decompression_raw: Arc<QTensor>,
    v_decompression_raw: Arc<QTensor>,
    o_proj: QMatMul,
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

        let q = qmatmul_forward(&self.q_compression, &x)?;
        let q = self.q_norm.forward(&q.to_dtype(DType::F32)?)?;
        let q = qmatmul_forward(&self.q_decompression, &q.to_dtype(compute_dtype)?)?;
        let q = q
            .reshape((b, s, self.n_heads, self.qk_head_dim))?
            .transpose(1, 2)?;

        // Split Query into NoPE and RoPE
        let (q_pass, q_rot) = (
            q.narrow(candle_core::D::Minus1, 0, self.qk_nope_head_dim)?,
            q.narrow(candle_core::D::Minus1, self.qk_nope_head_dim, self.rope_dim)?,
        );
        let q_rot_applied = self.rope.forward_interleaved(&q_rot, pos)?;
        let q_states =
            Tensor::cat(&[q_pass, q_rot_applied], candle_core::D::Minus1)?.contiguous()?;

        let compressed_kv = qmatmul_forward(&self.kv_compression, &x)?;
        let (kv_latent, k_rot) = (
            compressed_kv.narrow(candle_core::D::Minus1, 0, self.kv_lora_rank)?,
            compressed_kv.narrow(candle_core::D::Minus1, self.kv_lora_rank, self.rope_dim)?,
        );

        let kv_latent_norm = self
            .kv_norm
            .forward(&kv_latent.to_dtype(DType::F32)?.contiguous()?)?
            .to_dtype(compute_dtype)?;

        // Shapes: k_weights_raw = [20, 512, 192], v_weights_raw = [20, 256, 512]
        let k_weights_raw = self
            .k_decompression_raw
            .dequantize(kv_latent_norm.device())?;
        let v_weights_raw = self
            .v_decompression_raw
            .dequantize(kv_latent_norm.device())?;

        // 2. Add outer batch dimensions
        let k_weights_batched = k_weights_raw.unsqueeze(0)?; // [1, 20, 512, 192]
        let v_weights_batched = v_weights_raw.unsqueeze(0)?; // [1, 20, 256, 512]

        // 3. Broadcast hidden states across heads
        let kv_latent_norm_batched = kv_latent_norm.unsqueeze(1)?;
        let kv_latent_norm_broadcasted =
            kv_latent_norm_batched.broadcast_as((b, self.n_heads, s, 512))?;

        let k_nope = kv_latent_norm_broadcasted
            .matmul(&k_weights_batched)? // Results in [1, 20, S, 192]
            .contiguous()?;

        let value_states = kv_latent_norm_broadcasted
            .matmul(&v_weights_batched.transpose(2, 3)?)? // Results in [1, 20, S, 256]
            .contiguous()?;

        // Assemble the full Key (Key = K_nope + K_rope)
        let k_rot = k_rot
            .reshape((b, s, 1, self.rope_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k_rot_applied = self.rope.forward_interleaved(&k_rot, pos)?;
        let key_states = Tensor::cat(
            &[
                k_nope,
                k_rot_applied.broadcast_as((b, self.n_kv_heads, s, self.rope_dim))?,
            ],
            candle_core::D::Minus1,
        )?
        .contiguous()?;

        let (k_states, v_states) = update_cache(key_states, value_states, cache)?;
        let k_states = k_states.contiguous()?;
        let v_states = v_states.contiguous()?;

        profiling::range_push!("Compute attn");
        let attn =
            self.compute_attention(q_states, k_states, v_states, compute_dtype, use_flash)?;
        profiling::range_pop!();

        // Reshape and project output: [B, S, H, D] -> [B, S, hidden_size]
        let attn = attn
            .reshape((b, s, self.v_head_dim * self.n_heads))?
            .contiguous()?;
        qmatmul_forward(&self.o_proj, &attn)
    }

    fn compute_attention(
        &self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        compute_dtype: DType,
        #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] use_flash: bool,
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

                    let attn = self.compute_flash_attention(q, k, v, scale, compute_dtype)?;
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
            let mask = get_mask(
                &self.mask_indices,
                self.max_seq_len,
                seq_len,
                compute_dtype,
                q.device(),
            )?;
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
}

pub enum Deepseek2FfnMixer {
    Dense(DenseMixer),
    Moe(MoeMixer),
}

pub struct DenseMixer {
    up_proj: QMatMul,
    gate_proj: QMatMul,
    down_proj: QMatMul,
    compute_dtype: DType,
}

impl DenseMixer {
    fn forward(&self, input: &Tensor) -> anyhow::Result<Tensor> {
        let input = input.to_dtype(self.compute_dtype)?;

        // 1. Up and gate projections
        let up = qmatmul_forward(&self.up_proj, &input)?;
        let gate = qmatmul_forward(&self.gate_proj, &input)?;

        // 2. SwiGLU activation: silu(gate) * up (computed in F32 for stability)
        let gate_activated = silu(&gate.to_dtype(DType::F32)?)?;
        let gated = (gate_activated.to_dtype(self.compute_dtype)? * up)?;

        // 3. Down projection
        qmatmul_forward(&self.down_proj, &gated)
    }
}

pub struct MoeMixer {
    gate_inp: QMatMul,
    e_score_correction: Tensor,
    shared_mlp: DenseMixer,
    expert_up_raw: Arc<QTensor>,
    expert_gate_raw: Arc<QTensor>,
    expert_down_raw: Arc<QTensor>,
    n_experts: usize,           // 64
    n_expert_used: usize,       // 4 (top_k)
    n_group: usize,             // 1
    topk_group: usize,          // 1
    routed_scaling_factor: f64, // 1.8
    norm_top_k: bool,
    compute_dtype: DType,
}

impl MoeMixer {
    fn forward(&self, input: &Tensor) -> anyhow::Result<Tensor> {
        let (b, s, h) = input.dims3()?;

        // 1. Shared Expert Path (Expects 3D sequence inputs normally)
        let shared_out = self.shared_mlp.forward(input)?;

        // 2. CRITICAL FIX: Flatten 3D inputs to 2D matrix immediately
        // [Batch, Seq, Hidden] -> [Batch * Seq, Hidden]
        let x_flat = input.reshape((b * s, h))?.to_dtype(self.compute_dtype)?;

        // 3. Gate routing (Now safely receives a 2D matrix input)
        // Output will be 2D: [Batch * Seq, n_experts] -> [6, 64]
        let router_logits = self.gate_inp.forward(&x_flat)?;

        // 4. Evaluate Top-K selection routing
        // This will now find dims2() correctly without any rank crashes
        let (topk_indices, topk_weights) = self.route_tokens_to_experts(&router_logits)?;

        // 5. Execute hardware-accelerated MoE loops
        let final_routed = self.experts_forward(&x_flat, topk_indices, topk_weights)?;

        // 6. Combine tracks back into 3D geometry and return
        Ok((shared_out + final_routed.reshape((b, s, h)))?)
    }

    fn route_tokens_to_experts(&self, router_logits: &Tensor) -> anyhow::Result<(Tensor, Tensor)> {
        // Sigmoid and apply bias
        let router_logits = sigmoid(router_logits)?;
        let router_logits_for_choice = router_logits
            .broadcast_add(&self.e_score_correction)?
            .contiguous()?;

        // Group Scores Calculation
        let (tokens, _) = router_logits_for_choice.dims2()?;
        let experts_per_group = self.n_experts / self.n_group;
        let reshaped =
            router_logits_for_choice.reshape((tokens, self.n_group, experts_per_group))?;

        // Sort to get top 2 values
        let (sorted_group_experts, _) = reshaped.sort_last_dim(false)?; // Descending
        let group_scores = sorted_group_experts
            .narrow(candle_core::D::Minus1, 0, 2)? // Take top 2
            .sum(candle_core::D::Minus1)?; // Sum them

        // Group Index Selection
        let group_idx = group_scores
            .arg_sort_last_dim(false)? // Descending sort
            .narrow(candle_core::D::Minus1, 0, self.topk_group)?;

        // Create Group Mask using scatter_add
        let group_mask = Tensor::zeros_like(&group_scores)?.scatter_add(
            &group_idx,
            &Tensor::ones_like(&group_idx.to_dtype(group_scores.dtype())?)?,
            candle_core::D::Minus1,
        )?;

        // Expand Mask to Expert Score Mask
        let score_mask = group_mask
            .unsqueeze(candle_core::D::Minus1)?
            .expand((tokens, self.n_group, experts_per_group))?
            .reshape((tokens, self.n_experts))?;

        let score_mask_bool = score_mask.to_dtype(candle_core::DType::U8)?;

        let neg_inf_tensor = Tensor::full(
            f32::NEG_INFINITY,
            router_logits_for_choice.shape(),
            router_logits_for_choice.device(),
        )?
        .to_dtype(router_logits_for_choice.dtype())?;

        // Masked Fill
        let scores_for_choice =
            score_mask_bool.where_cond(&router_logits_for_choice, &neg_inf_tensor)?;

        // Get TopK Indices for Experts
        let topk_indices = scores_for_choice
            .arg_sort_last_dim(false)?
            .narrow(candle_core::D::Minus1, 0, self.n_expert_used)?
            .contiguous()?;

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

    fn experts_forward(
        &self,
        input: &Tensor,
        topk_indices: Tensor,
        topk_weights: Tensor,
    ) -> anyhow::Result<Tensor> {
        let (tokens, top_k) = topk_indices.dims2()?;
        let hidden_dim = input.dim(1)?;

        // 1. Temporarily dequantize ONLY this layer's 3D expert matrices
        // Weights are unpacked into standard float Tensors on the GPU instantly
        // Native shapes out of dequantize(): [2048, 1536, 64], [2048, 1536, 64], [1536, 2048, 64]
        let w_gate = self.expert_gate_raw.dequantize(input.device())?;
        let w_up = self.expert_up_raw.dequantize(input.device())?;
        let w_down = self.expert_down_raw.dequantize(input.device())?;

        // 2. Permute the weights so the Expert ID (64) is at Axis 0
        // Shape transformation: [Out, In, 64] -> [64, Out, In]
        // This is required to perform batch matrix multiplications across experts
        let w_gate = w_gate.permute((2, 0, 1))?; // Shape: [64, 2048, 1536]
        let w_up = w_up.permute((2, 0, 1))?; // Shape: [64, 2048, 1536]
        let w_down = w_down.permute((2, 0, 1))?; // Shape: [64, 1536, 2048]

        // 3. Prepare the Input Activations track
        // Repeat input top_k times to align with expert targets
        let dispatch_tokens = input
            .unsqueeze(1)?
            .repeat(vec![1, top_k, 1])?
            .flatten(0, 1)?; // Shape: [tokens * top_k, hidden_dim]

        let dispatch_expert_ids = topk_indices.flatten_all()?.contiguous()?; // Shape: [tokens * top_k]

        // 4. Gather the specific weight slices for our active tokens
        // Instead of computing all 64 experts, we use index_select to extract
        // ONLY the matrices for the experts chosen by the router
        let active_w_gate = w_gate.index_select(&dispatch_expert_ids, 0)?; // [tokens * top_k, 2048, 1536]
        let active_w_up = w_up.index_select(&dispatch_expert_ids, 0)?; // [tokens * top_k, 2048, 1536]
        let active_w_down = w_down.index_select(&dispatch_expert_ids, 0)?; // [tokens * top_k, 1536, 2048]

        // 5. Execute Fused Batched Multiplications
        // input: [N, 1, hidden_dim] x weight.transpose(): [N, hidden_dim, intermediate]
        let x_batched = dispatch_tokens.unsqueeze(1)?; // [tokens * top_k, 1, hidden_dim]

        let gate_out = x_batched
            .matmul(&active_w_gate.transpose(1, 2)?)?
            .squeeze(1)?; // [tokens * top_k, 2048]
        let up_out = x_batched
            .matmul(&active_w_up.transpose(1, 2)?)?
            .squeeze(1)?; // [tokens * top_k, 2048]

        // Apply SwiGLU Activation Function
        let current_hidden = (candle_nn::ops::silu(&gate_out)? * up_out)?; // [tokens * top_k, 2048]

        // Project Down
        let down_out = current_hidden
            .unsqueeze(1)?
            .matmul(&active_w_down.transpose(1, 2)?)?
            .squeeze(1)?; // [tokens * top_k, hidden_dim]

        // 6. Apply Routing Weights and Re-combine
        let weights_flat = topk_weights.flatten_all()?.unsqueeze(1)?;
        let weighted_hidden = down_out.broadcast_mul(&weights_flat)?;

        // Reshape back to [tokens, top_k, hidden_dim] and sum the experts track
        let combined_hidden = weighted_hidden
            .reshape((tokens, top_k, hidden_dim))?
            .sum(1)?; // Squeezes down to [tokens, hidden_dim]

        Ok(combined_hidden)
    }
}

impl Model for Deepseek2Model {
    fn init_cache(&self) -> anyhow::Result<Vec<Option<crate::model::MixedCache>>> {
        let caches = self
            .layers
            .iter()
            .map(|_layer| Some(MixedCache::KvCache(ConcatKvCache::new(2))))
            .collect();
        Ok(caches)
    }

    fn forward(
        &self,
        input: &Tensor,
        cache: &mut Vec<Option<MixedCache>>,
        pos: usize,
        use_flash: bool,
    ) -> anyhow::Result<Tensor> {
        // 1. Token embeddings
        profiling::range_push!("Embed step");
        let mut x = self
            .embed_layer
            .forward(input)?
            .to_dtype(self.compute_dtype)?;
        profiling::range_pop!();

        // 2. Transformer layers
        profiling::range_push!("Layer step");
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_span = trace_span!("layer", layer = i);
            x = layer_span.in_scope(|| {
                layer.forward(
                    &x,
                    cache.get_mut(i).unwrap(),
                    pos,
                    self.compute_dtype,
                    use_flash,
                )
            })?;
        }
        profiling::range_pop!();

        // 3. Final normalization and output projection
        let x_contiguous = x.to_dtype(DType::F32)?;
        let x = self.embed_norm.forward(&x_contiguous)?;
        let logits = qmatmul_forward(&self.lm_head, &x.to_dtype(self.compute_dtype)?)?;

        Ok(logits)
    }

    fn cache_seq_len_dim(&self) -> usize {
        self.cache_seq_len_dim
    }

    fn n_layers(&self) -> usize {
        self.layers.len()
    }
}

impl Deepseek2Model {
    pub fn load(
        config: ModelConfig,
        var_builder: VarBuilder,
        compute_dtype: DType,
        requested_max_seq_len: usize,
    ) -> anyhow::Result<Self> {
        profiling::range_push!("Deepseek2Model loading");
        debug!("Deepseek2 model load started...");
        let device = var_builder.device();
        let hidden_size = config.hidden_size;
        let n_heads = config.n_heads;
        let vocab_size = config.vocab_size;
        let rope_dim = config.n_rope_dims.unwrap_or(hidden_size / n_heads);

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
        debug!("Embedding loaded");
        // Load embedding normalization
        let embed_norm_prefix = find_norm_prefix(var_builder.clone());
        let embed_norm = RmsNorm::new(
            hidden_size,
            config.rms_epsilon as f64,
            var_builder.pp(embed_norm_prefix),
        )?;

        // Initialize rotary position embeddings
        let rope = Arc::new(RotaryEmbedding::new(
            config.rope_theta,
            rope_dim,
            effective_max_seq_len,
            true,
            device,
        )?);
        debug!("Rope created");
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
        debug!("Layers loaded");
        profiling::range_pop!();
        Ok(Self {
            embed_layer,
            layers,
            embed_norm,
            lm_head,
            compute_dtype,
            cache_seq_len_dim: config.cache_seq_len_dim,
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
            config,
            effective_max_seq_len,
        )?;

        let ffn_norm = RmsNorm::new(hidden_size, rms_epsilon, vb_l.pp("ffn_norm"))?;

        let ffn_mixer = Self::load_ffn_mixer(
            &vb_l,
            block_index,
            n_leading_dense_block,
            config,
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
    ) -> anyhow::Result<Deepseek2Attn> {
        let q_lora_rank = config.q_lora_rank.ok_or_else(|| {
            anyhow::anyhow!("Missing q_lora_rank, which is required for DeepSeek2 model")
        })?;
        let kv_lora_rank = config.kv_lora_rank.ok_or_else(|| {
            anyhow::anyhow!("Missing kv_lora_rank, which is required for DeepSeek2 model")
        })?;
        let attn_key_length_mla = config.attn_key_length_mla.ok_or_else(|| {
            anyhow::anyhow!("Missing attn_key_length_mla, which is required for DeepSeek2 model")
        })?;
        let attn_value_length_mla = config.attn_value_length_mla.ok_or_else(|| {
            anyhow::anyhow!("Missing attn_value_length_mla, which is required for DeepSeek2 model")
        })?;
        let n_kv_heads = config.n_kv_heads[block_index];
        let rms_epsilon = config.rms_epsilon as f64;

        Ok(Deepseek2Attn {
            q_compression: QMatMul::from_arc(var_builder.get_no_shape("attn_q_a.weight")?)?,
            q_norm: RmsNorm::new(q_lora_rank, rms_epsilon, var_builder.pp("attn_q_a_norm"))?,
            q_decompression: QMatMul::from_arc(var_builder.get_no_shape("attn_q_b.weight")?)?,
            kv_compression: QMatMul::from_arc(var_builder.get_no_shape("attn_kv_a_mqa.weight")?)?,
            kv_norm: RmsNorm::new(kv_lora_rank, rms_epsilon, var_builder.pp("attn_kv_a_norm"))?,
            k_decompression_raw: var_builder.get_no_shape("attn_k_b.weight")?,
            v_decompression_raw: var_builder.get_no_shape("attn_v_b.weight")?,
            o_proj: QMatMul::from_arc(var_builder.get_no_shape("attn_output.weight")?)?,
            rope,
            rope_dim,
            n_heads: config.n_heads,
            n_kv_heads,
            max_seq_len: effective_max_seq_len,
            mask_indices: OnceCell::new(),
            qk_head_dim: attn_key_length_mla,
            qk_nope_head_dim: attn_key_length_mla - rope_dim,
            v_head_dim: attn_value_length_mla,
            kv_lora_rank,
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

    fn load_dense_mixer(
        var_builder: &VarBuilder,
        compute_dtype: DType,
    ) -> anyhow::Result<DenseMixer> {
        Ok(DenseMixer {
            up_proj: QMatMul::from_arc(var_builder.get_no_shape("ffn_up.weight")?)?,
            gate_proj: QMatMul::from_arc(var_builder.get_no_shape("ffn_gate.weight")?)?,
            down_proj: QMatMul::from_arc(var_builder.get_no_shape("ffn_down.weight")?)?,
            compute_dtype,
        })
    }

    fn load_moe_mixer(
        var_builder: &VarBuilder,
        config: &ModelConfig,
        compute_dtype: DType,
    ) -> anyhow::Result<MoeMixer> {
        let n_experts = config.n_experts.ok_or_else(|| {
            anyhow::anyhow!("Missing n_experts, which is required for DeepSeek2 model")
        })?;
        let n_expert_used = config.n_expert_used.ok_or_else(|| {
            anyhow::anyhow!("Missing n_expert_used, which is required for DeepSeek2 model")
        })?;
        let n_group = config.n_expert_group.ok_or_else(|| {
            anyhow::anyhow!("Missing n_expert_group, which is required for DeepSeek2 model")
        })?;
        let topk_group = config.n_expert_group_used.ok_or_else(|| {
            anyhow::anyhow!("Missing n_expert_group_used, which is required for DeepSeek2 model")
        })?;
        let norm_top_k = config.expert_weights_norm.unwrap_or(false);
        let routed_scaling_factor = config.expert_weights_scale.ok_or_else(|| {
            anyhow::anyhow!("Missing expert_weights_scale, which is required for DeepSeek2 model")
        })? as f64;

        Ok(MoeMixer {
            gate_inp: QMatMul::from_arc(var_builder.get_no_shape("ffn_gate_inp.weight")?)?,
            e_score_correction: get_tensor("exp_probs_b.bias", var_builder.clone(), compute_dtype)?,
            shared_mlp: DenseMixer {
                up_proj: QMatMul::from_arc(var_builder.get_no_shape("ffn_up_shexp.weight")?)?,
                gate_proj: QMatMul::from_arc(var_builder.get_no_shape("ffn_gate_shexp.weight")?)?,
                down_proj: QMatMul::from_arc(var_builder.get_no_shape("ffn_down_shexp.weight")?)?,
                compute_dtype,
            },
            expert_up_raw: var_builder.get_no_shape("ffn_up_exps.weight")?,
            expert_gate_raw: var_builder.get_no_shape("ffn_gate_exps.weight")?,
            expert_down_raw: var_builder.get_no_shape("ffn_down_exps.weight")?,
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
