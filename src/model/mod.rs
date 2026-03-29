//! Model architectures and inference logic
//!
//! Implements the Llama/Qwen-style transformer forward pass with:
//! - RMSNorm, RoPE, multi-head attention, SwiGLU FFN
//! - CPU inference with f32 weights

use std::path::Path;

use crate::gguf::GgufFile;
use crate::ops;
use crate::quant::QuantizationType;
use crate::runtime::Backend;
use crate::tokenizer::Tokenizer;
use crate::weights::Weights;
use crate::Result;

/// Model configuration extracted from GGUF
#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub head_count: usize,
    pub head_count_kv: Option<usize>,
    pub head_dim: usize,
    pub rope_dim: Option<usize>,
    pub full_attention_interval: Option<usize>,
}

/// Complete model configuration
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arch: String,
    pub name: Option<String>,
    pub context_length: usize,
    pub embedding_length: usize,
    pub block_count: usize,
    pub attention: LayerConfig,
    pub feed_forward_length: usize,
    pub moe: Option<crate::gguf::MoeConfig>,
    pub quantization: QuantizationType,
    pub rms_epsilon: f32,
    pub rope_freq_base: f32,
}

/// KV cache for a single layer
struct LayerCache {
    key_cache: Vec<f32>,   // [seq_len * n_heads * head_dim]
    value_cache: Vec<f32>, // [seq_len * n_heads * head_dim]
}

impl LayerCache {
    fn new() -> Self {
        Self {
            key_cache: Vec::new(),
            value_cache: Vec::new(),
        }
    }

    fn seq_len(&self, head_dim: usize, n_heads: usize) -> usize {
        let per_step = n_heads * head_dim;
        if per_step == 0 {
            0
        } else {
            self.key_cache.len() / per_step
        }
    }
}

/// Inference context - main entry point for generation
pub struct InferenceContext {
    config: ModelConfig,
    backend: Box<dyn Backend>,
    weights: Weights,
    tokenizer: Tokenizer,
    layer_caches: Vec<LayerCache>,
}

impl InferenceContext {
    /// Build inference context from GGUF file
    pub fn from_gguf<P: AsRef<Path>>(gguf_path: P, backend: Box<dyn Backend>) -> Result<Self> {
        let gguf = GgufFile::open(&gguf_path)?;
        let config = gguf.model_config()?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        let weights = Weights::from_gguf(&gguf)?;

        let layer_caches = (0..config.block_count).map(|_| LayerCache::new()).collect();

        Ok(Self {
            config,
            backend,
            weights,
            tokenizer,
            layer_caches,
        })
    }

    /// Generate text given a prompt
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let input_ids = self.tokenizer.encode(prompt);

        if input_ids.is_empty() {
            return Ok(String::new());
        }

        let eos_id = self.tokenizer.eos_token_id();
        let mut generated = Vec::new();

        // Prefill: process all prompt tokens
        let mut logits = self.forward(&input_ids)?;

        // Decode loop
        for _ in 0..max_tokens {
            let next_token = sample_argmax(&logits);

            // Check for EOS
            if Some(next_token) == eos_id {
                break;
            }

            generated.push(next_token);

            // Forward pass with just the new token
            logits = self.forward(&[next_token])?;
        }

        let output = self.tokenizer.decode(&generated);
        Ok(output)
    }

    /// Transposed matvec: y = W^T @ x
    /// W is [n_rows, n_cols] row-major, x is [n_rows], y is [n_cols]
    fn matvec_t(y: &mut [f32], x: &[f32], w: &[f32], n_rows: usize, n_cols: usize) {
        for j in 0..n_cols {
            let mut sum = 0.0f32;
            for i in 0..n_rows {
                sum += x[i] * w[i * n_cols + j];
            }
            y[j] = sum;
        }
    }

    /// Single forward pass through the model
    ///
    /// Returns logits over the vocabulary for the last position.
    /// This is public for benchmarking and profiling use.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let n_embd = self.config.embedding_length;
        let n_layers = self.config.block_count;
        let n_heads = self.config.attention.head_count;
        let n_kv_heads = self.config.attention.head_count_kv.unwrap_or(n_heads);
        let head_dim = self.config.attention.head_dim;
        let n_ff = self.config.feed_forward_length;
        let vocab_size = self.tokenizer.vocab_size();
        let rope_dim = self.config.attention.rope_dim.unwrap_or(head_dim);
        let rms_eps = self.config.rms_epsilon;
        let rope_freq_base = self.config.rope_freq_base;

        // GGUF token_embd.weight: [n_embd, n_vocab] — each row is one embedding dim
        let embed_w = self.weights.get_data("token_embd.weight")?;

        // Derive vocab_size from embedding weight if tokenizer metadata is missing
        let tensor = self
            .weights
            .get("token_embd.weight")
            .ok_or_else(|| anyhow::anyhow!("token_embd.weight not found"))?;
        let actual_vocab = if tensor.shape.len() >= 2 {
            tensor.shape[1]
        } else {
            vocab_size
        };
        let eff_vocab = if vocab_size <= 10 {
            actual_vocab
        } else {
            vocab_size
        };
        let n_tokens = tokens.len();

        // Embedding lookup: embedding[i] = embed_w[i * eff_vocab + token_id]
        let mut x = vec![0.0f32; n_tokens * n_embd];
        for (t, &token_id) in tokens.iter().enumerate() {
            let id = token_id as usize;
            for i in 0..n_embd {
                let idx = i * eff_vocab + id;
                if idx < embed_w.len() {
                    x[t * n_embd + i] = embed_w[idx];
                }
            }
        }

        for layer_idx in 0..n_layers {
            let prefix = format!("blk.{layer_idx}");

            // Layer norm weights
            let attn_norm_w = self
                .weights
                .get_data(&format!("{prefix}.attn_norm.weight"))?;

            // Attention projection weights
            let wq = self.weights.get_data(&format!("{prefix}.attn_q.weight"))?;
            let wk = self.weights.get_data(&format!("{prefix}.attn_k.weight"))?;
            let wv = self.weights.get_data(&format!("{prefix}.attn_v.weight"))?;
            let wo = self
                .weights
                .get_data(&format!("{prefix}.attn_output.weight"))?;

            // FFN weights
            let ffn_gate = self
                .weights
                .get_data(&format!("{prefix}.ffn_gate.weight"))?;
            let ffn_up = self.weights.get_data(&format!("{prefix}.ffn_up.weight"))?;
            let ffn_down = self
                .weights
                .get_data(&format!("{prefix}.ffn_down.weight"))?;

            let cache = &mut self.layer_caches[layer_idx];
            let prev_seq_len = cache.seq_len(head_dim, n_kv_heads);

            // Process each position
            let mut layer_out = vec![0.0f32; n_tokens * n_embd];

            for pos in 0..n_tokens {
                let pos_offset = pos * n_embd;
                let mut normed = vec![0.0f32; n_embd];
                ops::rms_norm(
                    &mut normed,
                    &x[pos_offset..pos_offset + n_embd],
                    attn_norm_w,
                    rms_eps,
                );

                // Q, K, V projections
                let mut q = vec![0.0f32; n_heads * head_dim];
                let mut k = vec![0.0f32; n_kv_heads * head_dim];
                let mut v = vec![0.0f32; n_kv_heads * head_dim];

                ops::matvec(&mut q, &normed, wq, n_heads * head_dim, n_embd);
                ops::matvec(&mut k, &normed, wk, n_kv_heads * head_dim, n_embd);
                ops::matvec(&mut v, &normed, wv, n_kv_heads * head_dim, n_embd);

                // Apply RoPE
                let abs_pos = prev_seq_len + pos;
                for h in 0..n_heads {
                    let h_offset = h * head_dim;
                    let mut q_head = vec![0.0f32; head_dim];
                    q_head.copy_from_slice(&q[h_offset..h_offset + head_dim]);

                    let kv_h = h % n_kv_heads;
                    let kv_offset = kv_h * head_dim;
                    let mut k_head = vec![0.0f32; head_dim];
                    k_head.copy_from_slice(&k[kv_offset..kv_offset + head_dim]);

                    ops::apply_rope(
                        &mut q_head,
                        &mut k_head,
                        abs_pos,
                        head_dim,
                        rope_dim,
                        rope_freq_base,
                    );

                    q[h_offset..h_offset + head_dim].copy_from_slice(&q_head);
                    k[kv_offset..kv_offset + head_dim].copy_from_slice(&k_head);
                }

                // Append K, V to cache
                cache.key_cache.extend_from_slice(&k);
                cache.value_cache.extend_from_slice(&v);

                // Attention per head
                let total_seq_len = prev_seq_len + pos + 1;
                let mut attn_out = vec![0.0f32; n_heads * head_dim];

                for h in 0..n_heads {
                    let kv_h = h % n_kv_heads;
                    let q_offset = h * head_dim;
                    let mut head_out = vec![0.0f32; head_dim];

                    // Gather K, V for this head from cache
                    let mut k_head_cache = vec![0.0f32; total_seq_len * head_dim];
                    let mut v_head_cache = vec![0.0f32; total_seq_len * head_dim];

                    for t in 0..total_seq_len {
                        let kv_idx = t * n_kv_heads * head_dim + kv_h * head_dim;
                        k_head_cache[t * head_dim..(t + 1) * head_dim]
                            .copy_from_slice(&cache.key_cache[kv_idx..kv_idx + head_dim]);
                        v_head_cache[t * head_dim..(t + 1) * head_dim]
                            .copy_from_slice(&cache.value_cache[kv_idx..kv_idx + head_dim]);
                    }

                    ops::attention_head(
                        &mut head_out,
                        &q[q_offset..q_offset + head_dim],
                        &k_head_cache,
                        &v_head_cache,
                        total_seq_len,
                        head_dim,
                    );

                    attn_out[q_offset..q_offset + head_dim].copy_from_slice(&head_out);
                }

                // Output projection
                let mut proj_out = vec![0.0f32; n_embd];
                ops::matvec(&mut proj_out, &attn_out, wo, n_embd, n_heads * head_dim);

                // Residual connection
                ops::elem_add(
                    &mut layer_out[pos_offset..pos_offset + n_embd],
                    &x[pos_offset..pos_offset + n_embd],
                    &proj_out,
                );
            }

            // FFN with pre-norm
            let ffn_norm_w = self
                .weights
                .get_data(&format!("{prefix}.ffn_norm.weight"))?;
            for pos in 0..n_tokens {
                let pos_offset = pos * n_embd;
                let mut normed = vec![0.0f32; n_embd];
                ops::rms_norm(
                    &mut normed,
                    &layer_out[pos_offset..pos_offset + n_embd],
                    ffn_norm_w,
                    rms_eps,
                );

                let mut ffn_out = vec![0.0f32; n_embd];
                ops::ffn_swiglu(
                    &mut ffn_out,
                    &normed,
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                    n_embd,
                    n_ff,
                );

                // Residual
                let residual: Vec<f32> = layer_out[pos_offset..pos_offset + n_embd].to_vec();
                ops::elem_add(
                    &mut layer_out[pos_offset..pos_offset + n_embd],
                    &residual,
                    &ffn_out,
                );
            }

            x = layer_out;
        }

        // Final layer norm
        let output_norm_w = self.weights.get_data("output_norm.weight")?;
        let mut final_x = vec![0.0f32; n_embd];
        ops::rms_norm(
            &mut final_x,
            &x[(n_tokens - 1) * n_embd..],
            output_norm_w,
            rms_eps,
        );

        // Output projection (LM head)
        // Weight is [n_embd, n_vocab] — use transposed matvec
        let output_w = self
            .weights
            .get_data("output.weight")
            .or_else(|_| self.weights.get_data("token_embd.weight"))?;

        let mut logits = vec![0.0f32; eff_vocab];
        Self::matvec_t(&mut logits, &final_x, output_w, n_embd, eff_vocab);

        Ok(logits)
    }

    /// Get model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get backend information
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }

    /// Get tokenizer reference
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get weights reference
    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// Reset KV caches
    pub fn reset(&mut self) {
        for cache in &mut self.layer_caches {
            cache.key_cache.clear();
            cache.value_cache.clear();
        }
    }
}

/// Greedy sampling: pick the token with highest logit
fn sample_argmax(logits: &[f32]) -> u32 {
    let mut best_id = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &val) in logits.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_id = i as u32;
        }
    }
    best_id
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_model_config_parsing() {
        let path = "/home/jon/models/llama-cache/Arch-Agent-3B.Q8_0.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("SKIP: model not found");
            return;
        }
        let gguf = crate::gguf::GgufFile::open(path).expect("GGUF should load");
        let config = gguf.model_config().expect("config should parse");
        assert_eq!(config.arch, "qwen2");
        assert_eq!(config.block_count, 36);
        assert_eq!(config.embedding_length, 2048);
        assert_eq!(config.attention.head_count, 16);
        assert_eq!(config.attention.head_dim, 128);
    }
}
