use candle_core::Tensor;

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
