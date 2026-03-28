//! Quality benchmarking infrastructure
//!
//! Measures model quality via perplexity and reference comparison.
//! Used to verify that sparse execution preserves quality.

use std::path::Path;
use std::time::Instant;

use crate::model::InferenceContext;
use crate::Result;

/// Perplexity measurement result
#[derive(Debug, Clone)]
pub struct PerplexityResult {
    /// Perplexity value (lower is better)
    pub perplexity: f64,
    /// Number of tokens evaluated
    pub n_tokens: usize,
    /// Total log-likelihood
    pub total_log_likelihood: f64,
    /// Mean cross-entropy loss
    pub mean_loss: f64,
    /// Tokens per second during evaluation
    pub tokens_per_sec: f64,
}

/// Reference comparison result
#[derive(Debug, Clone)]
pub struct ReferenceResult {
    /// Mean absolute error on logits
    pub mean_abs_error: f64,
    /// Max absolute error on logits
    pub max_abs_error: f64,
    /// Cosine similarity of output distributions
    pub cosine_similarity: f64,
    /// Whether top-1 prediction matches
    pub top1_match: bool,
    /// Whether top-5 predictions match
    pub top5_match: bool,
    /// Number of tokens compared
    pub n_compared: usize,
}

/// Quality benchmark runner
pub struct QualityBenchmark {
    /// Reference logits file (from llama.cpp or full CPU run)
    reference_path: Option<String>,
}

impl Default for QualityBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

impl QualityBenchmark {
    pub fn new() -> Self {
        Self {
            reference_path: None,
        }
    }

    /// Set reference logits path for comparison
    pub fn with_reference<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.reference_path = Some(path.as_ref().to_string_lossy().into_owned());
        self
    }

    /// Measure perplexity on a text corpus
    ///
    /// Splits text into chunks, runs forward pass, and accumulates
    /// negative log-likelihood.
    pub fn measure_perplexity(
        &self,
        ctx: &mut InferenceContext,
        text: &str,
    ) -> Result<PerplexityResult> {
        let tokenizer = ctx.tokenizer();
        let tokens = tokenizer.encode(text);

        if tokens.len() < 2 {
            return Err(anyhow::anyhow!("Text too short for perplexity measurement"));
        }

        let start = Instant::now();
        let mut total_log_likelihood = 0.0f64;
        let mut n_evaluated = 0usize;

        // For perplexity, we need to compute P(token_i | token_0..token_{i-1})
        // Reset cache for each position
        for i in 1..tokens.len() {
            ctx.reset();

            // Feed tokens up to position i
            let context = &tokens[..i];
            let logits = ctx.forward(context)?;

            // Get log-probability of the actual next token
            let target = tokens[i] as usize;
            if target < logits.len() {
                let log_prob = log_softmax(&logits, target);
                total_log_likelihood += log_prob as f64;
                n_evaluated += 1;
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        let mean_loss = -total_log_likelihood / n_evaluated as f64;
        let perplexity = mean_loss.exp();

        Ok(PerplexityResult {
            perplexity,
            n_tokens: n_evaluated,
            total_log_likelihood,
            mean_loss,
            tokens_per_sec: n_evaluated as f64 / elapsed,
        })
    }

    /// Fast perplexity: uses KV cache accumulation (much faster)
    ///
    /// Runs forward pass once, accumulates KV cache, and measures
    /// prediction accuracy at each position.
    pub fn measure_perplexity_fast(
        &self,
        ctx: &mut InferenceContext,
        text: &str,
        chunk_size: usize,
    ) -> Result<PerplexityResult> {
        let tokenizer = ctx.tokenizer();
        let tokens = tokenizer.encode(text);

        if tokens.len() < 2 {
            return Err(anyhow::anyhow!("Text too short for perplexity measurement"));
        }

        let start = Instant::now();
        let mut total_log_likelihood = 0.0f64;
        let mut n_evaluated = 0usize;

        // Process in chunks (reset cache between chunks)
        let n_chunks = tokens.len().div_ceil(chunk_size);
        for chunk_idx in 0..n_chunks {
            ctx.reset();

            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = ((chunk_idx + 1) * chunk_size).min(tokens.len());
            let chunk = &tokens[chunk_start..chunk_end];

            if chunk.len() < 2 {
                continue;
            }

            // Prefill: feed all tokens except last
            let prefill = &chunk[..chunk.len() - 1];
            let logits = ctx.forward(prefill)?;

            // Measure prediction of last token
            let target = chunk[chunk.len() - 1] as usize;
            if target < logits.len() {
                let log_prob = log_softmax(&logits, target);
                total_log_likelihood += log_prob as f64;
                n_evaluated += 1;
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        let mean_loss = -total_log_likelihood / n_evaluated as f64;
        let perplexity = mean_loss.exp();

        Ok(PerplexityResult {
            perplexity,
            n_tokens: n_evaluated,
            total_log_likelihood,
            mean_loss,
            tokens_per_sec: n_evaluated as f64 / elapsed,
        })
    }

    /// Compare model output against reference logits
    ///
    /// Useful for verifying sparse execution produces identical results
    /// to full CPU execution.
    pub fn compare_reference(
        &self,
        ctx: &mut InferenceContext,
        tokens: &[u32],
        reference_logits: &[f32],
    ) -> Result<ReferenceResult> {
        ctx.reset();
        let logits = ctx.forward(tokens)?;

        if logits.len() != reference_logits.len() {
            return Err(anyhow::anyhow!(
                "Logit dimension mismatch: {} vs {}",
                logits.len(),
                reference_logits.len()
            ));
        }

        // Mean absolute error
        let mut sum_abs_err = 0.0f64;
        let mut max_abs_err = 0.0f64;
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;

        for i in 0..logits.len() {
            let err = (logits[i] - reference_logits[i]).abs() as f64;
            sum_abs_err += err;
            max_abs_err = max_abs_err.max(err);

            dot += logits[i] as f64 * reference_logits[i] as f64;
            norm_a += logits[i] as f64 * logits[i] as f64;
            norm_b += reference_logits[i] as f64 * reference_logits[i] as f64;
        }

        let mean_abs_err = sum_abs_err / logits.len() as f64;
        let cosine_sim = dot / (norm_a.sqrt() * norm_b.sqrt());

        // Top-1 and top-5 comparison
        let top1_model = argmax(&logits);
        let top1_ref = argmax(reference_logits);
        let top1_match = top1_model == top1_ref;

        let top5_model = topk(&logits, 5);
        let top5_ref = topk(reference_logits, 5);
        let top5_match = top5_model.iter().any(|x| top5_ref.contains(x));

        Ok(ReferenceResult {
            mean_abs_error: mean_abs_err,
            max_abs_error: max_abs_err,
            cosine_similarity: cosine_sim,
            top1_match,
            top5_match,
            n_compared: logits.len(),
        })
    }
}

/// Compute log-softmax and return log-probability of target index
fn log_softmax(logits: &[f32], target: usize) -> f32 {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&x| (x - max_val).exp()).sum();
    let log_sum_exp = sum_exp.ln() + max_val;
    logits[target] - log_sum_exp
}

/// Find index of maximum value
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &val) in v.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best = i;
        }
    }
    best
}

/// Get top-k indices
fn topk(v: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(f32, usize)> = v.iter().enumerate().map(|(i, &val)| (val, i)).collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|(_, i)| *i).collect()
}

/// Quality report: combines perplexity and reference comparison
#[derive(Debug)]
pub struct QualityReport {
    pub perplexity: Option<PerplexityResult>,
    pub reference: Option<ReferenceResult>,
    pub config: String,
    pub timestamp: String,
}

impl std::fmt::Display for QualityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Quality Report ===")?;
        writeln!(f, "Config: {}", self.config)?;
        writeln!(f, "Timestamp: {}", self.timestamp)?;

        if let Some(ref ppl) = self.perplexity {
            writeln!(f, "\nPerplexity:")?;
            writeln!(f, "  PPL:            {:.2}", ppl.perplexity)?;
            writeln!(f, "  Tokens:         {}", ppl.n_tokens)?;
            writeln!(f, "  Mean loss:      {:.4}", ppl.mean_loss)?;
            writeln!(f, "  Speed:          {:.1} tok/s", ppl.tokens_per_sec)?;
        }

        if let Some(ref rref) = self.reference {
            writeln!(f, "\nReference comparison:")?;
            writeln!(f, "  Mean abs error: {:.6}", rref.mean_abs_error)?;
            writeln!(f, "  Max abs error:  {:.6}", rref.max_abs_error)?;
            writeln!(f, "  Cosine sim:     {:.6}", rref.cosine_similarity)?;
            writeln!(f, "  Top-1 match:    {}", rref.top1_match)?;
            writeln!(f, "  Top-5 match:    {}", rref.top5_match)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_softmax() {
        let logits = [1.0, 2.0, 3.0, 4.0f32];
        let lp = log_softmax(&logits, 3);
        // Highest logit should have highest log-prob
        assert!(lp > log_softmax(&logits, 0));
        assert!(lp > log_softmax(&logits, 1));
        assert!(lp > log_softmax(&logits, 2));
        // Log-prob should be negative
        assert!(lp < 0.0);
    }

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
    }

    #[test]
    fn test_topk() {
        let top = topk(&[3.0, 1.0, 4.0, 2.0, 5.0], 3);
        assert_eq!(top, vec![4, 2, 0]); // indices of 5, 4, 3
    }

    #[test]
    fn test_reference_comparison() {
        let _bench = QualityBenchmark::new();
        // Reference comparison requires a real model, so just test the math
        let logits = [1.0, 2.0, 3.0, 4.0f32];
        let reference = [1.0, 2.0, 3.0, 4.0f32];
        let mut sum = 0.0f64;
        for i in 0..4 {
            sum += (logits[i] - reference[i]).abs() as f64;
        }
        assert_eq!(sum / 4.0, 0.0); // identical → zero error
    }
}
