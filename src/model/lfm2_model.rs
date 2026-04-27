use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "cuda")]
use candle_flash_attn::flash_attn;
use candle_nn::kv_cache::{self, ConcatKvCache};
use candle_nn::{
    Linear, Embedding,
    Conv1d, Conv1dConfig, Module,
    ops::{silu, softmax},
};
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::quantized_nn::{RmsNorm, linear_no_bias};
use candle_transformers::quantized_var_builder::VarBuilder;
use once_cell::sync::OnceCell;
use std::env::var;
use std::rc::Rc;

use crate::{
    model::{Model, model::MixedCache},
    model_config::ModelConfig,
};

pub struct Lfm2Model {
    pub embed_layer: Embedding, // token_embd.weight
    pub embed_norm: RmsNorm,    // token_embd_norm.weight (LFM2 specific)
    pub layers: Vec<Lfm2Layer>, // The 16 blocks
    pub lm_head: Linear,        // output.weight
    pub compute_dtype: DType,
}

pub struct Lfm2Layer {
    pub attn_norm: RmsNorm, // blk.N.attn_norm.weight (ALWAYS exists)
    pub mixer: Lfm2Mixer,   // The "Motor": either Attention or Conv
    pub ffn_norm: RmsNorm,  // blk.N.ffn_norm.weight
    pub ffn: Lfm2Ffn,       // gate, up, and down projections
    pub hidden_size: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub max_seq_len: usize,
}

impl Lfm2Layer {
    fn forward(
        &self,
        input: &Tensor,
        cache: &mut Option<MixedCache>,
        pos: usize,
        compute_dtype: DType,
        use_flash: bool,
    ) -> anyhow::Result<Tensor> {
        let pre_mixer_residual = input.clone();
        let x = self.attn_norm.forward(&input.to_dtype(DType::F32)?)?;  
        let mixed = match &self.mixer {
            Lfm2Mixer::Attention(attn_mixer) => attn_mixer.forward(
                &x,
                cache,
                pos,
                self.hidden_size,
                self.n_heads,
                self.n_kv_heads,
                self.max_seq_len,
                compute_dtype,
                use_flash,
            )?,
            Lfm2Mixer::ShortConv(conv_mixer) => conv_mixer.forward(&x, cache, compute_dtype)?,
        };
        
        let x = (mixed + pre_mixer_residual)?;
 
        let pre_mlp_residual = x.clone();
        let x = self.ffn_norm.forward(&x.to_dtype(DType::F32)?)?;
        
        let x = self.ffn.forward(&x.to_dtype(compute_dtype)?)?;
        
        let output = (x + pre_mlp_residual)?;
        
        anyhow::Ok(output)
    }
}

pub enum Lfm2Mixer {
    Attention(AttentionMixer),
    ShortConv(ShortConvMixer),
}

pub struct AttentionMixer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm, // blk.N.attn_q_norm
    k_norm: RmsNorm, // blk.N.attn_k_norm
    mask_indices: OnceCell<Tensor>,
    rope: Rc<RotaryEmbedding>,
}

impl AttentionMixer {
    pub fn forward(
        &self,
        x: &Tensor,
        cache: &mut Option<MixedCache>,
        pos: usize,
        hidden_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        max_seq_len: usize,
        compute_dtype: DType,
        use_flash: bool,
    ) -> anyhow::Result<Tensor> {
        let head_dim = hidden_size / n_heads;
        let x = x.to_dtype(compute_dtype)?;
        // Proj forward pass
        let q = self.q_proj.forward(&x)?;
        let k = self.k_proj.forward(&x)?;
        let v = self.v_proj.forward(&x)?;

        // Adding attention head dim
        let (b, s, _) = q.dims3()?;

        let q = q
            .reshape((b, s, n_heads, head_dim))?
            .permute((0, 2, 1, 3))?;
        let k = k
            .reshape((b, s, n_kv_heads, head_dim))?
            .permute((0, 2, 1, 3))?;
        let v = v
            .reshape((b, s, n_kv_heads, head_dim))?
            .permute((0, 2, 1, 3))?;

        // 1. Apply RoPE (Rotary Position Embeddings)
        let q = self.rope.forward(&q, pos)?;
        let k = self.rope.forward(&k, pos)?;
        
        // 2. Apply QK-Norm (LFM2 specific)
        let q = self.q_norm.forward(&q.to_dtype(DType::F32)?)?;
        let k = self.k_norm.forward(&k.to_dtype(DType::F32)?)?;

        // 3. Caching: Append current K,V to history in f16
        let k = k.to_dtype(DType::F16)?;
        let v = v.to_dtype(DType::F16)?;
        let (k, v) = match cache {
            Some(MixedCache::KvCache(kv_cache)) => {
                kv_cache.append(&k, &v).map_err(anyhow::Error::msg)?
            }
            _ => (k, v),
        };

        // 4. Attention Math
        let y = self.compute_attention(
            q,
            k,
            v,
            max_seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
            compute_dtype,
            use_flash,
        )?;

        // Flatten: [B, S, H, D] -> [B, S, H * D]
        let y = y.reshape((b, s, hidden_size))?;
        anyhow::Ok(self.o_proj.forward(&y)?)
    }

    fn compute_attention(
        &self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        max_seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        compute_dtype: DType,
        use_flash: bool,
    ) -> anyhow::Result<Tensor> {
        // Scores Q @ K.T) / sqrt(head_dim)
        let scale = 1.0 / (head_dim as f64).sqrt(); 

        #[cfg(feature = "cuda")]
        if use_flash && q.device().is_cuda() {
            // Flash attention expects [Batch, Seq, Heads, Head_Dim]
            // Note: LFM2 usually needs a transpose to put heads in the 3rd slot

            let q = q.transpose(1, 2)?.to_dtype(DType::BF16)?.contiguous()?;
            let k = k.transpose(1, 2)?.to_dtype(DType::BF16)?.contiguous()?;
            let v = v.transpose(1, 2)?.to_dtype(DType::BF16)?.contiguous()?;

            // causal: true handles the masking
            let attn = flash_attn(&q, &k, &v, scale as f32, true)?.to_dtype(compute_dtype)?;
            return anyhow::Ok(attn);
        }

        // Grouped-Query Attention in F32
        let (_b, _h, q_s, _d) = q.dims4()?;

        let k = crate::model::utility::repeat_kv(k, n_heads / n_kv_heads)?.to_dtype(DType::F32)?;
        let v = crate::model::utility::repeat_kv(v, n_heads / n_kv_heads)?.to_dtype(DType::F32)?;

        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        let attn = if q_s > 1 {
            let mask = self.get_mask(q_s, max_seq_len, compute_dtype, q.device())?;
            let attn = attn.to_dtype(compute_dtype)?;
            attn.broadcast_add(&mask)?.to_dtype(DType::F32)?
        } else {
            attn
        };

        let attn_weights = softmax(&attn, candle_core::D::Minus1)?;

        let y_t = attn_weights.matmul(&v)?;

        // Cast back to compute dtype
        let y_t = y_t.to_dtype(compute_dtype)?;

        anyhow::Ok(y_t.transpose(1, 2)?)
    }

    fn get_mask(
        &self,
        seq_len: usize,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        // Slice the 1D indices to [S]
        let indices = self
            .mask_indices
            .get_or_try_init(|| Tensor::arange(0u32, max_seq_len as u32, device))?;
        let idx = indices.narrow(0, 0, seq_len)?;

        // Broadcast into [S, 1] and [1, S]
        let i = idx.reshape((seq_len, 1))?;
        let j = idx.reshape((1, seq_len))?;

        // Broadcast [S, 1] to [S, S] and [1, S] to [S, S]
        let i = i.broadcast_as((seq_len, seq_len))?;
        let j = j.broadcast_as((seq_len, seq_len))?;

        let cond = i.lt(&j)?; // This is [S, S]

        let on_true = Tensor::new(f32::NEG_INFINITY, device)?
            .to_dtype(dtype)?
            .broadcast_as((seq_len, seq_len))?; // Match the shape!

        let on_false = Tensor::new(0f32, device)?
            .to_dtype(dtype)?
            .broadcast_as((seq_len, seq_len))?; // Match the shape!

        anyhow::Ok(cond.where_cond(&on_true, &on_false)?)

    }
}

pub struct ShortConvMixer {
    in_proj: Linear,  // Input, Gate, and Condition projections
    conv: Conv1d,     // The depth-wise 1D convolution
    out_proj: Linear, // Output projection
    conv_l_cache: usize,
    kernel_size: usize,
}

impl ShortConvMixer {
    pub fn forward(
        &self,
        input: &Tensor,
        cache: &mut Option<MixedCache>,
        compute_dtype: DType,
    ) -> anyhow::Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = input.dims3()?;

        // 1. Project: [B, S, 1024] -> [B, S, 3072]
        let in_proj = self.in_proj.forward(&input.to_dtype(compute_dtype)?)?;

        // 2. Split into 3 chunks of 1024
        let chunks = in_proj.chunk(3, 2)?;
        let (b_gate, c_gate, h_signal) = (chunks[0].clone(), chunks[1].clone(), chunks[2].clone());

        // 3. First Gating: y = B ⊙ h˜
        let y = (b_gate * h_signal)?; 

        // 3. Convolution: z = Convk(y)
        let y_t = y.transpose(1, 2)?;

        // 4. Add cache or padding if needed and perform conv forward
        let conv_out = if seq_len == 1 {
            if let Some(MixedCache::ConvCache(state)) = cache {
                let y_cached = Tensor::cat(&[state.as_ref(), &y_t], 2)?;
                let s_current = y_cached.dims()[2];
                let y_narrowed =
                    y_cached.narrow(2, s_current - self.conv_l_cache, self.conv_l_cache)?;
                *cache = Some(MixedCache::ConvCache(y_cached.narrow(
                    2,
                    s_current - (self.conv_l_cache - 1),
                    self.conv_l_cache - 1,
                )?));
                let y_narrowed = y_narrowed.to_dtype(DType::F32)?;
                self.conv.forward(&y_narrowed)?

            } else {
                // Single token without cache: pad left to match kernel size
                let pad_len = self.kernel_size - 1;
                let pad = Tensor::zeros((batch_size, hidden_size, pad_len), y_t.dtype(), y_t.device())?;
                let y_padded = Tensor::cat(&[pad, y_t.clone()], 2)?;
                let y_padded = y_padded.to_dtype(DType::F32)?;
                self.conv.forward(&y_padded)?
            }
        } else {
            // PRE-FILL
            // 1. Manually pad left: [B, 1024, 2]
            let pad_len = self.kernel_size - 1;
            let pad = Tensor::zeros((batch_size, hidden_size, pad_len), y_t.dtype(), y_t.device())?;
            let y_padded = Tensor::cat(&[pad, y_t.clone()], 2)?;

            // 2. Update cache with the last 2 tokens of the REAL signal
            if seq_len >= pad_len {
                *cache = Some(MixedCache::ConvCache(y_t.narrow(2, seq_len - pad_len, pad_len)?));
            }

            // 3. Forward: Result length is (S + 2) - 3 + 1 = S.
            let y_padded = y_padded.to_dtype(DType::F32)?;
            self.conv.forward(&y_padded)?
        };

        let conv_out = conv_out.to_dtype(compute_dtype)?;
        // Move back to [B, S, 1024]
        let conv_out = conv_out.transpose(1, 2)?;

        // 4. Second Gating: C ⊙ z
        let gated_z = (c_gate * conv_out)?;

        // 5. Output Projection: o = Linearout(...)
        anyhow::Ok(self.out_proj.forward(&gated_z)?)
    }
}

pub struct Lfm2Ffn {
    pub up_proj: Linear,   // blk.N.ffn_up
    pub gate_proj: Linear, // blk.N.ffn_gate
    pub down_proj: Linear, // blk.N.ffn_down
    pub compute_dtype: DType,
}

impl Lfm2Ffn {
    pub fn forward(&self, input: &Tensor) -> anyhow::Result<Tensor> {
        // 1. Projections in compute_dtype (F16/BF16)
        let input = &input.to_dtype(self.compute_dtype)?;
        let up_proj = self.up_proj.forward(input)?;
        let gate_proj = self.gate_proj.forward(input)?;

        // 2. Activation in F32 for stability
        let gate_f32 = gate_proj.to_dtype(DType::F32)?;
        let activated = candle_nn::ops::silu(&gate_f32)?;
        
        // 3. Multiply and cast back to compute_dtype
        let x = (activated.to_dtype(self.compute_dtype)? * up_proj)?;

        // 4. Final projection in compute_dtype
        anyhow::Ok(self.down_proj.forward(&x)?)
    }
}

impl Model for Lfm2Model {
    fn forward(
        &self,
        input: &Tensor,
        cache: &mut Vec<Option<MixedCache>>,
        pos: usize,
        use_flash: bool,
    ) -> anyhow::Result<Tensor> {
        // 1. Embeddings
        let mut x = input.to_dtype(self.compute_dtype)?;
        x = self.embed_layer.forward(input)?;

        // 2. Attention and conv layers
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, cache.get_mut(i).unwrap(), pos, self.compute_dtype, use_flash)?;
        }

        // 3. Embed norm
        let output = self.embed_norm.forward(&x.to_dtype(DType::F32)?)?;

        // 4. Output projection
        let logits = self.lm_head.forward(&output.to_dtype(self.compute_dtype)?)?;
        
        anyhow::Ok(logits)
    }

    fn init_cache(&self) -> anyhow::Result<Vec<Option<MixedCache>>> {
        let mut caches = Vec::new();
        for layer in &self.layers {
            match &layer.mixer {
                Lfm2Mixer::Attention(_) => {
                    caches.push(Some(MixedCache::KvCache(ConcatKvCache::new(2))))
                }
                Lfm2Mixer::ShortConv(_) => caches.push(None),
            }
        }
        Ok(caches)
    }
}

impl Lfm2Model {
    pub fn load(config: ModelConfig, var_builder: VarBuilder, compute_dtype: DType) -> anyhow::Result<Self> {
        let rms_epsilon = config.rms_epsilon as f64;
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;
        let n_layers = config.n_layers;
        let n_heads = config.n_heads;
        let intermediate_ffn_size = config.intermediate_ffn_size;
        let max_seq_len = config.max_seq_len;
        let head_dim = hidden_size / n_heads;
        let device = var_builder.device();

        // Retrieve embd weights
        let embed_tensor = Self::get_tensor(
            "token_embd.weight", 
            var_builder.clone(), 
            compute_dtype
        )?;

        // Final output head
        let lm_head_tensor = if var_builder.contains_key("output.weight") {
            Self::get_tensor("output.weight", var_builder.clone(), compute_dtype)?
        } else {
            embed_tensor.clone()
        };

        let embed_layer = Embedding::new(embed_tensor, hidden_size);
        let lm_head = Linear::new(lm_head_tensor, None);

        // Retrieve embd norm weights
        let embed_norm = RmsNorm::new(hidden_size, rms_epsilon, var_builder.pp("token_embd_norm"))?;

        // Init Rope
        let rope = Rc::new(RotaryEmbedding::new(
            config.rope_theta,
            head_dim,
            max_seq_len,
            device,
        )?);

        //Build layers
        let mut layers: Vec<Lfm2Layer> = vec![];
        let rope = Rc::clone(&rope);

        for i in 0..n_layers {
            let vb_l = var_builder.pp(format!("blk.{}", i));
            // 1. Attention norm
            let attn_norm = RmsNorm::new(hidden_size, rms_epsilon, vb_l.pp("attn_norm"))?;
            let n_kv_heads = config.n_kv_heads[i];

            // 2. Mixer of attention or shortconv
            let mixer = if n_kv_heads > 0 {
                let kv_proj_out_dim = (n_kv_heads * hidden_size) / n_heads;
                let attn_mixer = AttentionMixer {
                    q_proj: Linear::new(
                        Self::get_tensor(
                            "attn_q.weight", 
                            vb_l.clone(), 
                            compute_dtype
                        )?, 
                        None),
                    k_proj: Linear::new(
                        Self::get_tensor(
                            "attn_k.weight", 
                            vb_l.clone(), 
                            compute_dtype
                        )?, 
                        None),
                    v_proj: Linear::new(
                        Self::get_tensor(
                            "attn_v.weight", 
                            vb_l.clone(), 
                            compute_dtype
                        )?, 
                        None),
                    o_proj: Linear::new(
                        Self::get_tensor(
                            "attn_output.weight", 
                            vb_l.clone(), 
                            compute_dtype
                        )?, 
                        None), 
                    q_norm: RmsNorm::new(head_dim, rms_epsilon, vb_l.pp("attn_q_norm"))?,
                    k_norm: RmsNorm::new(head_dim, rms_epsilon, vb_l.pp("attn_k_norm"))?,
                    rope: rope.clone(),
                    mask_indices: OnceCell::new(),
                };
                Lfm2Mixer::Attention(attn_mixer)
            } else {
                let raw_conv_weights = vb_l.get_no_shape("shortconv.conv.weight")?;
                let (channels, kernel_size) = raw_conv_weights.shape().dims2()?;
                let conv_cfg = Conv1dConfig {
                    groups: channels,
                    padding: 0,
                    ..Default::default()
                };
                let conv_weights = raw_conv_weights
                    .dequantize(device)?
                    .unsqueeze(1)?;
                let conv_mixer = ShortConvMixer {
                    in_proj: Linear::new(
                        Self::get_tensor(
                            "shortconv.in_proj.weight", 
                            vb_l.clone(), 
                            compute_dtype)?, 
                        None),
                    conv: Conv1d::new(conv_weights, None, conv_cfg),
                    out_proj: Linear::new(
                        Self::get_tensor(
                            "shortconv.out_proj.weight", 
                            vb_l.clone(), 
                            compute_dtype)?, 
                        None),
                    conv_l_cache: config.conv_l_cache,
                    kernel_size,
                };
                Lfm2Mixer::ShortConv(conv_mixer)
            };

            // 3. ffn norm
            let ffn_norm = RmsNorm::new(hidden_size, rms_epsilon, vb_l.pp("ffn_norm"))?;

            // 4. Common ffn block
            let ffn = Lfm2Ffn {
                up_proj: Linear::new(
                        Self::get_tensor(
                            "ffn_up.weight", 
                            vb_l.clone(), 
                            compute_dtype)?, 
                        None),
                gate_proj: Linear::new(
                        Self::get_tensor(
                            "ffn_gate.weight", 
                            vb_l.clone(), 
                            compute_dtype)?, 
                        None),
                down_proj: Linear::new(
                        Self::get_tensor(
                            "ffn_down.weight", 
                            vb_l.clone(), 
                            compute_dtype)?, 
                        None),
                compute_dtype,
            };

            // 5. Append layer
            layers.push(Lfm2Layer {
                attn_norm,
                mixer,
                ffn_norm,
                ffn,
                hidden_size,
                n_heads,
                n_kv_heads,
                max_seq_len,
            });
        }

        anyhow::Ok(Self {
            embed_layer,
            embed_norm,
            layers,
            lm_head,
            compute_dtype,
        })
    }

    fn get_tensor(tensor_name: &str, var_builder: VarBuilder, compute_dtype: DType) -> anyhow::Result<Tensor> {
        let q_tensor = var_builder.get_no_shape(tensor_name)?;
        let device = var_builder.device();
        
        let weight = match (q_tensor.dtype(), compute_dtype) {
            // If the target is F32, always use the standard dequantize
            (_, DType::F32) => q_tensor.dequantize(device)?,

            // If the source is already a float (F16/BF16), just dequantize + cast
            (GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::F32, _) => {
                q_tensor.dequantize(device)?.to_dtype(compute_dtype)?
            }

            // If source is quantized and target is F16, use the fast path
            (_, DType::F16) => q_tensor.dequantize_f16(device)?,

            // For any other target (like BF16), dequantize to F32 then cast
            _ => q_tensor.dequantize(device)?.to_dtype(compute_dtype)?,
        };
        
        anyhow::Ok(weight)
    }
}

pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        rope_theta: f32,
        head_dim: usize,
        max_seq_len: usize,
        device: &Device,
    ) -> anyhow::Result<Self> {
        // 1. Generate the inverse frequencies (theta)
        // theta_i = 1.0 / (base ^ (2i / dim))
        let freqs: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / (rope_theta as f64).powf(i as f64 / head_dim as f64))
            .collect();
        let freqs = Tensor::new(freqs, device)?.to_dtype(candle_core::DType::F32)?;

        // 2. Create the position range [0, 1, 2, ..., max_seq_len]
        let t =
            Tensor::arange(0u32, max_seq_len as u32, device)?.to_dtype(candle_core::DType::F32)?;

        // 3. Compute outer product: [max_seq_len, 1] * [1, head_dim/2] -> [max_seq_len, head_dim/2]
        let freqs = t
            .reshape((max_seq_len, 1))?
            .matmul(&freqs.reshape((1, freqs.dims1()?))?)?;

        // 4. Cat with itself to match head_dim
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;

        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    pub fn forward(&self, x: &Tensor, pos: usize) -> anyhow::Result<Tensor> {
        // 1. Store the original dtype to convert back later
        let original_dtype = x.dtype();
        let (_b, _h, seq_len, head_dim) = x.dims4()?;

        // 2. Slice and prepare sin/cos (already in f32)
        let cos = self
            .cos
            .narrow(0, pos, seq_len)?
            .reshape((1, 1, seq_len, head_dim))?;
        let sin = self
            .sin
            .narrow(0, pos, seq_len)?
            .reshape((1, 1, seq_len, head_dim))?;

        // 3. Cast x to f32 for the rotation math
        let x_f32 = x.to_dtype(candle_core::DType::F32)?;

        // Standard RoPE rotation: x_rotated = x*cos + rotate_half(x)*sin
        let last_dim = x_f32.dims().last().unwrap();
        let x1 = x_f32.narrow(candle_core::D::Minus1, 0, last_dim / 2)?;
        let x2 = x_f32.narrow(candle_core::D::Minus1, last_dim / 2, last_dim / 2)?;
        // rotate_half([x1, x2]) -> [-x2, x1]
        let x2_f32 = Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)?;

        // 4. Perform math in f32
        let rotated_f32 = (x_f32.broadcast_mul(&cos)? + x2_f32.broadcast_mul(&sin)?)?;

        // 5. Cast back to original dtype (e.g., f16, bf16, or f32)
        anyhow::Ok(
            rotated_f32
                .to_dtype(original_dtype)
                .map_err(anyhow::Error::msg)?,
        )
    }
}
