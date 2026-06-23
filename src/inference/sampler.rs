use candle_core::{Result, Tensor};
use rand::distr::{Distribution, weighted::WeightedIndex};
use std::collections::HashMap;
use crate::config::GenerationConfig;
use crate::types::{ToolChoiceMode, ToolChoice};

#[derive(Debug, Clone)]
pub struct SamplingResult {
    pub token: u32,
    pub logprob: Option<f32>,
    pub top_k_logprobs: Option<Vec<TopKEntry>>,
}

#[derive(Debug, Clone, Copy)]
pub struct TopKEntry {
    pub token_id: u32,
    pub logprob: f32,
}
pub trait Sampler {
    fn sample(&mut self, logits: &Tensor, top_k_logprobs: Option<usize>) -> anyhow::Result<SamplingResult>;
    fn add_token(&mut self, token_id: u32);
}

pub struct InferenceSampler {
    special_token_ids: Vec<u32>,
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    is_tool_call_forbidden: bool,
    tool_start_token_id: Option<usize>,
    rng: rand::rngs::StdRng,
    logits_buffer: Vec<f32>,
    sort_buffer: Vec<(usize, f32)>,
    history_window: Vec<u32>,
    history_head: usize,
}

impl InferenceSampler {
    /// Configures a new standalone logic processor
    pub fn new(config: GenerationConfig, special_token_ids: Vec<u32>, vocab_size: usize, penalty_window: usize, tool_start_token_id: Option<u32>) -> Self {
        use rand::SeedableRng;
        Self {
            special_token_ids,
            temperature: config.temperature.max(0.0),
            top_p: config.top_p.clamp(0.0, 1.0),
            repetition_penalty: config.repetition_penalty.max(1.0),
            is_tool_call_forbidden: config.tool_choice == ToolChoice::Mode(ToolChoiceMode::None) && tool_start_token_id.is_some(),
            tool_start_token_id: tool_start_token_id.and_then(|v| Some(v as usize)),
            rng: {
                let seed_bytes = (config.seed as u128).to_le_bytes();
                let mut rng_seed = [0u8; 32];
                rng_seed[..16].copy_from_slice(&seed_bytes);
                rng_seed[16..].copy_from_slice(&seed_bytes);
                rand::SeedableRng::from_seed(rng_seed)
            },
            // Pre-allocate the full vocabulary size upfront
            logits_buffer: vec![0.0; vocab_size],
            sort_buffer: vec![(0, 0.0); vocab_size],
            // Fixed-size window tracking prevents infinite scaling slowdowns
            history_window: vec![0; penalty_window],
            history_head: 0,
        }
    }

    /// Records generated tokens to accurately calculate loop penalties
    fn record_token(&mut self, token_id: u32) {
        if self.history_window.is_empty() || self.special_token_ids.contains(&token_id) { 
            return; 
        }
        self.history_window[self.history_head] = token_id;
        self.history_head = (self.history_head + 1) % self.history_window.len();
    }

    /// Primary entry point: transforms raw logits and samples a single token ID
    #[tracing::instrument(level = "debug", skip_all)]
    fn sample_impl(&mut self, logits: &Tensor, top_k_logprobs: Option<usize>) -> anyhow::Result<SamplingResult> {
        // 1. Ensure logits are flattened to a 1D vector representing the final prediction step
        let logits_tensor = if logits.rank() == 1 {
            logits.clone()
        } else {
            logits.flatten_all()?  
        };

        let cpu_logits = logits_tensor
            .to_dtype(candle_core::DType::F32)?
            .to_device(&candle_core::Device::Cpu)?;

        let (storage, layout) = cpu_logits.storage_and_layout();
        
        if let candle_core::Storage::Cpu(cpu_storage) = &*storage {
            let entire_raw_slice: &[f32] = cpu_storage.as_slice()?;
            let start_offset = layout.start_offset();
            let end_offset = start_offset + cpu_logits.shape().elem_count();
            let raw_slice = &entire_raw_slice[start_offset..end_offset];
            self.logits_buffer[..raw_slice.len()].copy_from_slice(raw_slice);
        } else {
            anyhow::bail!("Failed to resolve CPU storage mapping");
        }
        let vocab_len = self.logits_buffer.len();

        // 2. Apply Repetition Penalty (Calculated before Softmax)
        if self.repetition_penalty > 1.0 {
            for &token_id in &self.history_window {
                let idx = token_id as usize;
                if idx < vocab_len {
                    let val = self.logits_buffer[idx];
                    if val > 0.0 {
                        self.logits_buffer[idx] = val / self.repetition_penalty;
                    } else {
                        self.logits_buffer[idx] = val * self.repetition_penalty;
                    }
                }
            }
        }

        // Ban Tool Call
        if self.is_tool_call_forbidden {
            if let Some(tool_start_token_id) = self.tool_start_token_id {
                if tool_start_token_id < vocab_len {
                    self.logits_buffer[tool_start_token_id] = f32::NEG_INFINITY;
                }
            }
        }

        // 3. Handle Greedy Sampling (If Temp is 0, pick highest score directly)
        if self.temperature == 0.0 {
            let sampled = self.logits_buffer.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap();
            self.record_token(sampled);

            return Ok(SamplingResult {
                token: sampled,
                logprob: None,
                top_k_logprobs: None,
            });
        }

        // 4. Apply Temperature Scaling
        let inv_temp = 1.0 / self.temperature;
        let mut max_logit = f32::NEG_INFINITY;
        for i in 0..vocab_len {
            self.logits_buffer[i] *= inv_temp;
            if self.logits_buffer[i] > max_logit {
                max_logit = self.logits_buffer[i];
            }
        }

        let mut sum_exponents = 0.0;
        for i in 0..vocab_len {
            let exp_val = (self.logits_buffer[i] - max_logit).exp();
            // Temporarily use sort_buffer to cache raw exponent numbers safely
            self.sort_buffer[i] = (i, exp_val); 
            sum_exponents += exp_val;
        }

        // 5. Normalize probabilities in place and copy into sorting scratchpad
        let inv_sum = 1.0 / sum_exponents;
        for i in 0..vocab_len {
            let true_prob = self.sort_buffer[i].1 * inv_sum;
            self.logits_buffer[i] = true_prob;
            // Overwrite sort_buffer with true normalized states for the Top-P step
            self.sort_buffer[i] = (i, true_prob); 
        }

        // 6. Apply Top-P
        if self.top_p < 1.0 {
            self.sort_buffer.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let mut cumulative_sum = 0.0;
            let mut cut_off = vocab_len;
            for i in 0..vocab_len {
                cumulative_sum += self.sort_buffer[i].1;
                if cumulative_sum >= self.top_p {
                    cut_off = i + 1;
                    break;
                }
            }

            // Zero out sampling probabilities for rejected tokens
            let inv_remaining = 1.0 / cumulative_sum;
            for i in 0..cut_off {
                self.logits_buffer[self.sort_buffer[i].0] = self.sort_buffer[i].1 * inv_remaining;
            }
            for i in cut_off..vocab_len {
                self.logits_buffer[self.sort_buffer[i].0] = 0.0;
            }
        }

        // 7. Weighted Random Distribution Selection
        let sampled_token = match WeightedIndex::new(&self.logits_buffer) {
            Ok(dist) => dist.sample(&mut self.rng) as u32,
            Err(_) => {
                // All probabilities zeroed out (top-p filtering or numerical underflow).
                // Fall back to uniform sampling across the vocab.
                use rand::distr::Uniform;
                let range = Uniform::new(0, vocab_len).expect("vocab_len > 0");
                range.sample(&mut self.rng) as u32
            }
        };
        
        self.record_token(sampled_token);

        // 8. Calculate Logprobs for the Top-K tokens
        let (logprob, top_k_logprobs) = if let Some(top_k_logprobs) = top_k_logprobs {
            let top_k = top_k_logprobs.clamp(0, 5);
            if self.top_p >= 1.0 {
                self.sort_buffer.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            }

            let chosen_true_prob = self.sort_buffer.iter()
                .find(|&&(idx, _)| idx == sampled_token as usize)
                .map(|&(_, prob)| prob)
                .unwrap_or(1e-45);

            let mut entries = Vec::with_capacity(top_k);
            for i in 0..top_k {
                let (token_idx, prob) = self.sort_buffer[i];
                let safe_prob = if prob > 0.0 { prob } else { 1e-45 };
                entries.push(TopKEntry {
                    token_id: token_idx as u32,
                    logprob: safe_prob.ln(),
                });
            }
            (Some(chosen_true_prob.ln()), Some(entries))
        } else {
            (None, None)
        };

        Ok(SamplingResult {
            token: sampled_token,
            logprob,
            top_k_logprobs,
        })
    }
}

impl Sampler for InferenceSampler {
    fn sample(&mut self, logits: &Tensor, top_k_logprobs: Option<usize>) -> anyhow::Result<SamplingResult> {
        self.sample_impl(logits, top_k_logprobs)
    }

    fn add_token(&mut self, token_id: u32) {
        self.record_token(token_id)
    }
}
