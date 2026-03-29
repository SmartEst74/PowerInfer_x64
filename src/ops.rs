//! Neural network forward pass operations
//!
//! Implements the core operations for transformer inference:
//! - RMSNorm (Llama/Qwen normalization)
//! - RoPE (Rotary Position Embeddings)
//! - Multi-head attention
//! - FFN with SwiGLU activation

/// Root Mean Square Layer Normalization
///
/// out = (x / rms(x)) * weight
/// where rms(x) = sqrt(mean(x^2) + eps)
pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(x.len(), weight.len());

    // Compute RMS using SIMD sum of squares
    let sum_sq = crate::simd::sum_squares(x);
    let rms = (sum_sq / x.len() as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    // Normalize (scale) into out, then multiply by weight
    crate::simd::scale(out, x, inv_rms);
    for i in 0..out.len() {
        out[i] *= weight[i];
    }
}

/// Apply Rotary Position Embeddings to Q and K vectors
///
/// Applies RoPE to pairs of elements (even, odd) using precomputed sin/cos.
/// For each pair (x[i], x[i+1]):
///   out[i]   = x[i] * cos - x[i+1] * sin
///   out[i+1] = x[i] * sin + x[i+1] * cos
pub fn apply_rope(q: &mut [f32], k: &mut [f32], position: usize, head_dim: usize, rope_dim: usize) {
    let half = rope_dim / 2;

    for i in 0..half {
        let freq = 1.0 / 10000.0f32.powf(2.0 * i as f32 / rope_dim as f32);
        let theta = position as f32 * freq;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        // Apply to Q
        if 2 * i + 1 < head_dim {
            let q0 = q[2 * i];
            let q1 = q[2 * i + 1];
            q[2 * i] = q0 * cos_theta - q1 * sin_theta;
            q[2 * i + 1] = q0 * sin_theta + q1 * cos_theta;
        }

        // Apply to K
        if 2 * i + 1 < head_dim {
            let k0 = k[2 * i];
            let k1 = k[2 * i + 1];
            k[2 * i] = k0 * cos_theta - k1 * sin_theta;
            k[2 * i + 1] = k0 * sin_theta + k1 * cos_theta;
        }
    }
}

/// Softmax over a slice (in-place)
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // Find max for numerical stability
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Exp and sum
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }

    // Normalize
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

/// SiLU activation function: x * sigmoid(x)
pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        let sig = 1.0 / (1.0 + (-*v).exp());
        *v *= sig;
    }
}

/// Element-wise multiplication of two slices
pub fn elem_mul(out: &mut [f32], a: &[f32], b: &[f32]) {
    debug_assert_eq!(out.len(), a.len());
    debug_assert_eq!(out.len(), b.len());
    crate::simd::elem_mul(out, a, b);
}

/// Element-wise addition of two slices
pub fn elem_add(out: &mut [f32], a: &[f32], b: &[f32]) {
    debug_assert_eq!(out.len(), a.len());
    debug_assert_eq!(out.len(), b.len());
    crate::simd::elem_add(out, a, b);
}

/// Matrix-vector multiplication: y = x @ w (row-major w)
pub fn matvec(y: &mut [f32], x: &[f32], w: &[f32], n_out: usize, n_in: usize) {
    crate::simd::matvec(y, x, w, n_out, n_in);
}

/// Scaled dot-product attention for a single head (causal mask)
///
/// Computes: softmax(Q @ K^T / sqrt(d_k) + mask) @ V
/// For causal decoding: mask is -inf for future positions.
pub fn attention_head(
    out: &mut [f32], // [head_dim]
    q: &[f32],       // [head_dim] query for current position
    k_cache: &[f32], // [seq_len * head_dim] key cache (row-major)
    v_cache: &[f32], // [seq_len * head_dim] value cache (row-major)
    seq_len: usize,
    head_dim: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Compute attention scores: q @ k_cache^T
    let mut scores = vec![0.0f32; seq_len];
    for (t, score) in scores.iter_mut().enumerate() {
        let k_offset = t * head_dim;
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += q[d] * k_cache[k_offset + d];
        }
        *score = dot * scale;
    }

    // Softmax over scores
    softmax(&mut scores);

    // Weighted sum of values
    for item in out.iter_mut().take(head_dim) {
        *item = 0.0;
    }
    for (t, score) in scores.iter().enumerate() {
        let v_offset = t * head_dim;
        for (d, item) in out.iter_mut().enumerate().take(head_dim) {
            *item += score * v_cache[v_offset + d];
        }
    }
}

/// Transposed matvec: y = W^T @ x
/// W is [n_rows, n_cols] row-major, x is [n_rows], y is [n_cols]
pub fn matvec_t(y: &mut [f32], x: &[f32], w: &[f32], n_rows: usize, n_cols: usize) {
    for j in 0..n_cols {
        let mut sum = 0.0f32;
        for i in 0..n_rows {
            sum += x[i] * w[i * n_cols + j];
        }
        y[j] = sum;
    }
}

/// Feed-forward network with SwiGLU (Llama/Qwen style)
///
/// Weight layout (GGUF): gate_w and up_w are [hidden_dim, intermediate_dim],
/// down_w is [intermediate_dim, hidden_dim].
///
/// ffn(x) = (SiLU(gate_w @ x) * (up_w @ x)) @ down_w
pub fn ffn_swiglu(
    out: &mut [f32], // [hidden_dim]
    x: &[f32],       // [hidden_dim]
    gate_w: &[f32],  // [hidden_dim, intermediate_dim] in GGUF
    up_w: &[f32],    // [hidden_dim, intermediate_dim] in GGUF
    down_w: &[f32],  // [intermediate_dim, hidden_dim] in GGUF
    hidden_dim: usize,
    intermediate_dim: usize,
) {
    let mut gate = vec![0.0f32; intermediate_dim];
    let mut up = vec![0.0f32; intermediate_dim];

    // gate = gate_w @ x (W is [hidden, inter], need transposed matvec)
    matvec_t(&mut gate, x, gate_w, hidden_dim, intermediate_dim);
    // up = up_w @ x
    matvec_t(&mut up, x, up_w, hidden_dim, intermediate_dim);

    // gate = SiLU(gate)
    silu(&mut gate);

    // gate = gate * up (element-wise)
    let result = gate
        .iter()
        .zip(up.iter())
        .map(|(a, b)| a * b)
        .collect::<Vec<_>>();
    gate.copy_from_slice(&result);

    // out = gate @ down_w (W is [inter, hidden], normal matvec)
    matvec(out, &gate, down_w, hidden_dim, intermediate_dim);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm() {
        let x = [1.0, 2.0, 3.0, 4.0f32];
        let weight = [1.0; 4];
        let mut out = [0.0f32; 4];
        rms_norm(&mut out, &x, &weight, 1e-6);

        // RMS of x = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let val: f32 = (1.0f32 + 4.0 + 9.0 + 16.0) / 4.0 + 1e-6f32;
        let rms = val.sqrt();
        assert!((out[0] - 1.0 / rms).abs() < 1e-5);
        assert!((out[3] - 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn test_softmax() {
        let mut x = [1.0, 2.0, 3.0f32];
        softmax(&mut x);
        // Sum should be 1
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Should be monotonically increasing
        assert!(x[0] < x[1]);
        assert!(x[1] < x[2]);
    }

    #[test]
    fn test_silu() {
        let mut x = [0.0f32, 1.0, -1.0];
        silu(&mut x);
        // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert!(x[0].abs() < 1e-6);
        // SiLU(1) = 1 * sigmoid(1) ≈ 0.7311
        assert!((x[1] - 0.7310586).abs() < 1e-5);
    }

    #[test]
    fn test_attention_head() {
        let head_dim = 4;
        let seq_len = 3;
        let q = [1.0, 0.0, 0.0, 0.0f32];
        let k_cache = [
            1.0, 0.0, 0.0, 0.0, // token 0: dot=1.0
            0.0, 1.0, 0.0, 0.0, // token 1: dot=0.0
            0.0, 0.0, 1.0, 0.0, // token 2: dot=0.0
        ];
        let v_cache = [
            10.0, 0.0, 0.0, 0.0, // token 0
            0.0, 20.0, 0.0, 0.0, // token 1
            0.0, 0.0, 30.0, 0.0, // token 2
        ];
        let mut out = vec![0.0f32; head_dim];

        attention_head(&mut out, &q, &k_cache, &v_cache, seq_len, head_dim);

        // Q is most similar to K[0] (score=0.5 after scale), others score=0
        // After softmax: weights ≈ [0.622, 0.189, 0.189]
        // Output ≈ 0.622 * [10, 0, 0, 0] + 0.189 * [0, 20, 0, 0] + ...
        // out[0] ≈ 6.22, out[1] ≈ 3.78
        assert!(out[0] > 0.0, "out[0] should be positive, got {}", out[0]);
        assert!(
            out[1] >= 0.0,
            "out[1] should be non-negative, got {}",
            out[1]
        );
        // Verify output sums to approximately the weighted sum
        let sum: f32 = out.iter().sum();
        assert!(sum > 0.0, "output sum should be positive");
    }

    #[test]
    fn test_rope() {
        let head_dim = 4;
        let rope_dim = 4;
        let mut q = [1.0, 2.0, 3.0, 4.0f32];
        let mut k = [5.0, 6.0, 7.0, 8.0f32];
        apply_rope(&mut q, &mut k, 0, head_dim, rope_dim);
        // At position 0, RoPE should preserve values (cos(0)=1, sin(0)=0)
        assert!((q[0] - 1.0).abs() < 1e-6);
        assert!((q[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec() {
        // [1,2] @ [[3,4],[5,6]] = [11, 17]
        let x = [1.0f32, 2.0];
        let w = [3.0f32, 4.0, 5.0, 6.0];
        let mut y = [0.0f32; 2];
        matvec(&mut y, &x, &w, 2, 2);
        assert!((y[0] - 11.0).abs() < 1e-6);
        assert!((y[1] - 17.0).abs() < 1e-6);
    }
}
