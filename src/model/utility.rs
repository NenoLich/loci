use candle_core::{Tensor, Device, DType, Module};
use candle_transformers::quantized_var_builder::VarBuilder;
use candle_core::quantized::{GgmlDType, QMatMul};

pub fn repeat_kv(x: Tensor, n_rep: usize) -> anyhow::Result<Tensor> {
    if n_rep == 1 {
        anyhow::Ok(x)
    } else {
        let (batch_size, n_kv_heads, seq_len, head_dim) = x.dims4()?;

        let y = x
            .unsqueeze(2)?
            .expand((batch_size, n_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((batch_size, n_kv_heads * n_rep, seq_len, head_dim))?;

        anyhow::Ok(y)
    }
}

pub fn qmatmul_forward(qmatmul: &QMatMul, xs: &Tensor) -> anyhow::Result<Tensor> {
    Ok(match xs.dtype() {
        DType::F16 => qmatmul.forward_via_f16(xs)?,
        DType::F32 => qmatmul.forward(xs)?,
        _ => anyhow::bail!("Unsupported dtype: {:?} for qmatmul forward operation", xs.dtype()),
    })
}

/// Load a tensor from the GGUF model and convert to the target compute dtype.
///
/// Handles quantized tensors (Q4, Q8, etc.) and converts them to the
/// appropriate compute dtype (F16, BF16, or F32) with optimized paths.
pub fn get_tensor(
    tensor_name: &str,
    var_builder: VarBuilder,
    compute_dtype: DType,
) -> anyhow::Result<Tensor> {
    let q_tensor = var_builder.get_no_shape(tensor_name)?;
    let device = var_builder.device();

    let weight = match (q_tensor.dtype(), compute_dtype) {
        // Target is F32: always dequantize to F32
        (_, DType::F32) => q_tensor.dequantize(device)?,

        // Source is already float: dequantize and cast
        (GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::F32, _) => {
            q_tensor.dequantize(device)?.to_dtype(compute_dtype)?
        }

        // Quantized source to F16: use fast dequantize path
        (_, DType::F16) => q_tensor.dequantize_f16(device)?,

        // All other cases: dequantize to F32 then cast
        _ => q_tensor.dequantize(device)?.to_dtype(compute_dtype)?,
    };

    Ok(weight)
}

pub fn find_norm_prefix(vb: VarBuilder) -> String {
    let candidates = ["output_norm", "token_embd_norm", "model.norm"];
    for &c in &candidates {
        // GGUF weights usually end in .weight
        if vb.contains_key(&format!("{}.weight", c)) {
            return c.to_string();
        }
    }
    "output_norm".to_string() // Fallback
}

pub fn nonzero_1d(t: &Tensor) -> anyhow::Result<Tensor> {
    let device = t.device();
    // Convert tensor to a flat vector of floats to find indices on CPU
    let data = t.flatten_all()?.to_vec1::<f32>()?;
    
    let indices: Vec<u32> = data.iter()
        .enumerate()
        .filter(|&(_, &val)| val != 0.0)
        .map(|(idx, _)| idx as u32)
        .collect();

    if indices.is_empty() {
        // Return an empty tensor of the correct dtype for indexing
        Ok(Tensor::from_slice(&[] as &[u32], (0,), device)?)
    } else {
        Ok(Tensor::from_vec(indices.clone(), (indices.len(),), device)?)
    }
}

/// Rotary Position Embeddings (RoPE) for LFM2.
///
/// RoPE encodes position information by rotating query and key vectors
/// in the complex plane. This implementation precomputes sine and cosine
/// tables for all positions up to `max_seq_len`.
pub struct RotaryEmbedding {
    /// Precomputed cosine values for all positions
    cos: Tensor,
    /// Precomputed sine values for all positions
    sin: Tensor,
}

impl RotaryEmbedding {
    /// Create a new RotaryEmbedding instance.
    ///
    /// # Arguments
    /// * `rope_theta` - Base frequency for RoPE (typically 10000.0)
    /// * `head_dim` - Dimension of each attention head
    /// * `max_seq_len` - Maximum sequence length to precompute
    /// * `device` - Device to store the precomputed tables
    pub fn new(
        rope_theta: f32,
        head_dim: usize,
        max_seq_len: usize,
        interleaved: bool, // New flag
        device: &Device,
    ) -> anyhow::Result<Self> {
        let freqs: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / (rope_theta as f64).powf(i as f64 / head_dim as f64))
            .collect();
        let freqs = Tensor::new(freqs, device)?.to_dtype(DType::F32)?;
        let positions = Tensor::arange(0u32, max_seq_len as u32, device)?.to_dtype(DType::F32)?;
        let freqs = positions.reshape((max_seq_len, 1))?.matmul(&freqs.reshape((1, ()))?)?;

        let freqs = if interleaved {
            // Pattern: [f1, f1, f2, f2...]
            freqs.unsqueeze(2)?.repeat(vec![1, 1, 2])?.reshape((max_seq_len, head_dim))?
        } else {
            // Pattern: [f1, f2, f1, f2...]
            Tensor::cat(&[&freqs, &freqs], 1)?
        };

        Ok(Self { cos: freqs.cos()?, sin: freqs.sin()? })
    }


    /// Apply rotary position embeddings to a tensor.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, heads, seq_len, head_dim]
    /// * `pos` - Starting position in the sequence
    ///
    /// # Returns
    /// Tensor with RoPE applied, same shape as input
    pub fn forward(&self, x: &Tensor, pos: usize) -> anyhow::Result<Tensor> {
        let original_dtype = x.dtype();
        let (_batch, _heads, seq_len, head_dim) = x.dims4()?;

        // Slice sine and cosine for the current position range
        let cos = self
            .cos
            .narrow(0, pos, seq_len)?
            .reshape((1, 1, seq_len, head_dim))?;
        let sin = self
            .sin
            .narrow(0, pos, seq_len)?
            .reshape((1, 1, seq_len, head_dim))?;

        // Convert to F32 for rotation math
        let x_f32 = x.to_dtype(candle_core::DType::F32)?;

        // Standard RoPE: x_rotated = x*cos + rotate_half(x)*sin
        // rotate_half([x1, x2]) -> [-x2, x1]
        let half_dim = head_dim / 2;
        let x1 = x_f32.narrow(candle_core::D::Minus1, 0, half_dim)?;
        let x2 = x_f32.narrow(candle_core::D::Minus1, half_dim, half_dim)?;
        let x_rotated = Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)?;

        // Apply rotation: x*cos + rotated_x*sin
        let rotated = (x_f32.broadcast_mul(&cos)? + x_rotated.broadcast_mul(&sin)?)?;

        // Cast back to original dtype
        rotated
            .to_dtype(original_dtype)
            .map_err(anyhow::Error::msg)
    }

    pub fn forward_interleaved(&self, x: &Tensor, pos: usize) -> anyhow::Result<Tensor> {
        let original_dtype = x.dtype();
        let (b, h, seq_len, head_dim) = x.dims4()?;

        // 1. Prepare Cos/Sin
        let cos = self.cos.narrow(0, pos, seq_len)?
            .reshape((1, 1, seq_len, head_dim))?;
        let sin = self.sin.narrow(0, pos, seq_len)?
            .reshape((1, 1, seq_len, head_dim))?;

        let x_f32 = x.to_dtype(candle_core::DType::F32)?;

        // 2. Interleaved rotate_half:
        // We want to transform [x0, x1, x2, x3] into [-x1, x0, -x3, x2]
        
        // Reshape to group pairs: [b, h, s, head_dim/2, 2]
        let x_pairs = x_f32.reshape((b, h, seq_len, head_dim / 2, 2))?;
        
        // Slice x0 and x1 from the last dimension
        let x0 = x_pairs.narrow(candle_core::D::Minus1, 0, 1)?;
        let x1 = x_pairs.narrow(candle_core::D::Minus1, 1, 1)?;
        
        // Create interleaved rotated: concat([-x1, x0]) and flatten back
        let x_rotated = Tensor::cat(&[&x1.neg()?, &x0], candle_core::D::Minus1)?
            .reshape((b, h, seq_len, head_dim))?;

        // 3. Apply rotation
        let rotated = (x_f32.broadcast_mul(&cos)? + x_rotated.broadcast_mul(&sin)?)?;

        rotated.to_dtype(original_dtype).map_err(anyhow::Error::msg)
    }

}

