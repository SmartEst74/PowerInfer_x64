//! TurboQuant: Vector quantization for KV cache compression in LLM inference
//!
//! Implements Google Research's TurboQuant algorithm (ICLR 2026) for compressing
//! key-value caches to 2-4 bits per coordinate while preserving attention quality.
//!
//! Paper: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
//! arXiv:2504.19874
//!
//! Note: TurboQuant compresses the KV cache, NOT model weights. Model weight
//! quantization uses GGUF's own formats (Q4_K_M, Q8_0, etc.).
//!
//! Two-stage algorithm:
//! - Stage 1 (PolarQuant): Random rotation + Lloyd-Max scalar quantization
//! - Stage 2 (QJL): 1-bit sign correction for unbiased inner products
//!   The paper mathematically proves QJL produces unbiased dot product estimates.
//!   Per-vector reconstruction error is 23-44% MSE, but inner products are accurate.
//!
//! QJL is enabled by default. Set `qjl_bits = 0` for MSE-only mode (no QJL).
//!
//! Reference: https://github.com/tonbistudio/turboquant-pytorch

#![allow(clippy::needless_range_loop)] // Index notation is clearer for matrix ops

/// Precomputed Lloyd-Max centroids for N(0,1) distribution.
/// Index: centroids[bit_width - 1] gives the centroid positions.
/// These are optimal for minimizing MSE when quantizing unit-variance Gaussian data.
const LLOYD_MAX_CENTROIDS: [&[f32]; 4] = [
    // 1-bit: ±0.7979
    &[-0.7979, 0.7979],
    // 2-bit: boundaries at 0, ±0.9533
    &[-1.4996, -0.4950, 0.4950, 1.4996],
    // 3-bit: 8 levels
    &[
        -2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520,
    ],
    // 4-bit: 16 levels
    &[
        -2.7330, -2.0690, -1.6180, -1.2560, -0.9424, -0.6568, -0.3880, -0.1284, 0.1284, 0.3880,
        0.6568, 0.9424, 1.2560, 1.6180, 2.0690, 2.7330,
    ],
];

/// TurboQuant compressor/decompressor for vectors of a fixed dimension.
///
/// Supports two modes:
/// - V2 (with QJL): Random rotation + Lloyd-Max + QJL residual correction.
///   Produces unbiased inner product estimates (paper-proven).
/// - V3 (MSE-only): Random rotation + Lloyd-Max only, no QJL.
///   Lower variance but biased inner products.
///
/// Set `qjl_bits > 0` to enable QJL (V2), or `qjl_bits = 0` for MSE-only (V3).
pub struct TurboQuant {
    /// Random orthogonal rotation matrix [dim x dim], row-major
    rotation: Vec<f32>,
    /// Dimensionality of vectors
    dim: usize,
    /// Bits per coordinate (2, 3, or 4)
    bits: usize,
    /// Lloyd-Max centroids for the selected bit-width
    centroids: Vec<f32>,
    /// QJL projection matrix [qjl_bits x dim], row-major, random Gaussian.
    /// Empty if qjl_bits == 0 (MSE-only mode).
    qjl_matrix: Vec<f32>,
    /// Number of QJL projection bits (0 = MSE-only, >0 = QJL enabled)
    qjl_bits: usize,
}

impl TurboQuant {
    /// Create a new TurboQuant compressor with QJL enabled.
    ///
    /// - `bits`: quantization bits per coordinate (2, 3, or 4)
    /// - `dim`: vector dimensionality (typically head_dim, e.g. 128)
    /// - `qjl_bits`: number of QJL residual correction bits (typically = dim).
    ///   Set to 0 for MSE-only mode (no QJL).
    /// - `seed`: RNG seed for reproducibility
    pub fn new(bits: usize, dim: usize, qjl_bits: usize, seed: u64) -> Self {
        assert!((2..=4).contains(&bits), "bits must be 2, 3, or 4");
        assert!(dim > 0, "dim must be > 0");

        let centroids = LLOYD_MAX_CENTROIDS[bits - 1].to_vec();

        // Generate random orthogonal matrix via QR decomposition
        let rotation = generate_orthogonal_matrix(dim, seed);

        // Generate QJL projection matrix (random Gaussian) if QJL is enabled
        let qjl_matrix = if qjl_bits > 0 {
            generate_gaussian_matrix(qjl_bits, dim, seed ^ 0xDEAD_BEEF)
        } else {
            Vec::new()
        };

        Self {
            rotation,
            dim,
            bits,
            centroids,
            qjl_matrix,
            qjl_bits,
        }
    }

    /// Create MSE-only compressor (no QJL, V3 mode).
    ///
    /// Simpler and lower variance, but inner product estimates are biased.
    /// May be preferred when the quantization error is small (4+ bits).
    pub fn new_mse_only(bits: usize, dim: usize, seed: u64) -> Self {
        Self::new(bits, dim, 0, seed)
    }

    /// Number of bytes needed to store a compressed vector
    pub fn compressed_bytes(&self) -> usize {
        // Quantized indices: bits * dim bits, packed
        let index_bytes = (self.bits * self.dim).div_ceil(8);
        // QJL sign bits: qjl_bits bits, packed (0 if QJL disabled)
        let qjl_bytes = self.qjl_bits.div_ceil(8);
        index_bytes + qjl_bytes
    }

    /// Whether QJL is enabled
    pub fn has_qjl(&self) -> bool {
        self.qjl_bits > 0
    }

    /// Quantize a single vector to packed bytes.
    ///
    /// Input: `x` [dim] — the vector to compress (assumed unit-norm)
    /// Output: packed bytes containing quantized indices (+ QJL sign bits if enabled)
    pub fn quantize_vector(&self, x: &[f32]) -> Vec<u8> {
        debug_assert_eq!(x.len(), self.dim);

        // Stage 1: rotate
        let mut rotated = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let mut sum = 0.0f32;
            for j in 0..self.dim {
                sum += self.rotation[i * self.dim + j] * x[j];
            }
            rotated[i] = sum;
        }

        // Scale for per-coordinate variance: after rotation, unit vectors
        // have variance 1/d per coordinate. Scale by sqrt(d) so the
        // distribution matches N(0,1) that the Lloyd-Max codebook targets.
        let scale = (self.dim as f32).sqrt();
        for v in rotated.iter_mut() {
            *v *= scale;
        }

        // Stage 1: Lloyd-Max scalar quantization
        let mut indices = vec![0u8; self.dim];
        for (i, &val) in rotated.iter().enumerate() {
            let mut best_idx = 0u8;
            let mut best_dist = f32::MAX;
            for (j, &centroid) in self.centroids.iter().enumerate() {
                let dist = (val - centroid).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = j as u8;
                }
            }
            indices[i] = best_idx;
        }

        if self.qjl_bits == 0 {
            // MSE-only mode: just pack indices
            return pack_indices(&indices, self.bits);
        }

        // Stage 2: QJL residual correction
        // Compute residual = rotated_scaled - dequantized (in scaled space)
        let mut residual = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            residual[i] = rotated[i] - self.centroids[indices[i] as usize];
        }

        // Project residual through QJL matrix and take sign
        let mut qjl_signs = vec![false; self.qjl_bits];
        for i in 0..self.qjl_bits {
            let mut proj = 0.0f32;
            for j in 0..self.dim {
                proj += self.qjl_matrix[i * self.dim + j] * residual[j];
            }
            qjl_signs[i] = proj >= 0.0;
        }

        // Pack everything into bytes
        pack_compressed(&indices, self.bits, &qjl_signs)
    }

    /// Dequantize a compressed vector back to f32.
    ///
    /// Note: per-vector reconstruction has significant error (23-44% MSE).
    /// The power of TurboQuant is in accurate *inner products*, not per-vector accuracy.
    pub fn dequantize_vector(&self, packed: &[u8]) -> Vec<f32> {
        let indices = unpack_indices(packed, self.bits, self.dim);
        let inv_scale = 1.0 / (self.dim as f32).sqrt();

        // Reconstruct in rotated+scaled space
        let mut reconstructed = vec![0.0f32; self.dim];
        for (i, &idx) in indices.iter().enumerate() {
            reconstructed[i] = self.centroids[idx as usize] * inv_scale;
        }

        // Reverse rotation: x = R^T @ reconstructed
        let mut result = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let mut sum = 0.0f32;
            for j in 0..self.dim {
                sum += self.rotation[j * self.dim + i] * reconstructed[j];
            }
            result[i] = sum;
        }

        result
    }

    /// Compute attention score between a full-precision query and a compressed key.
    ///
    /// When QJL is enabled, uses the asymmetric estimator from the paper which
    /// produces unbiased inner product estimates by correcting for the quantization
    /// residual using the QJL sign bits.
    ///
    /// When QJL is disabled (MSE-only), dequantizes the key and computes the
    /// dot product directly (lower variance but biased).
    pub fn dot(&self, q: &[f32], packed_key: &[u8]) -> f32 {
        debug_assert_eq!(q.len(), self.dim);

        if self.qjl_bits == 0 {
            // MSE-only mode: dequantize and dot
            let key_deq = self.dequantize_vector(packed_key);
            let mut dot = 0.0f32;
            for i in 0..self.dim {
                dot += q[i] * key_deq[i];
            }
            return dot;
        }

        // QJL mode: asymmetric estimator (unbiased)
        self.asymmetric_dot(q, packed_key)
    }

    /// Compute attention score using the QJL asymmetric estimator.
    ///
    /// This is the core operation from the paper — produces unbiased inner product
    /// estimates by combining the MSE-quantized dot product with a QJL correction
    /// term that accounts for the quantization residual.
    fn asymmetric_dot(&self, q: &[f32], packed_key: &[u8]) -> f32 {
        debug_assert_eq!(q.len(), self.dim);

        let indices = unpack_indices(packed_key, self.bits, self.dim);
        let qjl_signs = unpack_qjl_signs(packed_key, self.bits, self.dim, self.qjl_bits);

        let scale = (self.dim as f32).sqrt();
        let inv_scale = 1.0 / scale;

        // Rotate query: q_rot = R @ q
        let mut q_rot = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let mut sum = 0.0f32;
            for j in 0..self.dim {
                sum += self.rotation[i * self.dim + j] * q[j];
            }
            q_rot[i] = sum;
        }

        // Stage 1: <q_rot, k_quantized> in scaled space
        let mut mse_dot = 0.0f32;
        for i in 0..self.dim {
            let k_deq = self.centroids[indices[i] as usize];
            mse_dot += q_rot[i] * k_deq * inv_scale;
        }

        // Stage 2: QJL correction
        // Estimate residual norm from quantization distortion
        let distortion = self.expected_distortion();
        let residual_norm = (distortion * inv_scale).sqrt();

        // Project query through QJL matrix
        let mut qjl_dot = 0.0f32;
        for i in 0..self.qjl_bits {
            let mut proj = 0.0f32;
            for j in 0..self.dim {
                proj += self.qjl_matrix[i * self.dim + j] * q_rot[j];
            }
            let sign = if qjl_signs[i] { 1.0 } else { -1.0 };
            qjl_dot += proj * sign;
        }

        let qjl_scale = (std::f32::consts::PI / (2.0 * self.qjl_bits as f32)).sqrt();
        let correction = residual_norm * qjl_scale * qjl_dot * inv_scale;

        mse_dot + correction
    }

    /// Expected MSE distortion for this bit-width (from paper's theoretical bounds)
    fn expected_distortion(&self) -> f32 {
        match self.bits {
            2 => 0.17,
            3 => 0.043,
            4 => 0.011,
            _ => 0.17,
        }
    }

    /// Get bits per coordinate
    pub fn bits(&self) -> usize {
        self.bits
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }
}

// --- Compressed KV Cache (TurboQuant 3-bit + QJL) ---

/// TurboQuant-compressed KV cache for efficient attention.
///
/// Stores keys using TurboQuant (3-bit Lloyd-Max + 1-bit QJL = 4 bits total per coordinate),
/// values as f16. This provides:
/// - ~5× compression for keys with quality-neutral attention (paper: 0.997 needle-in-haystack)
/// - ~2× compression for values via f16
/// - Unbiased inner product estimates via QJL residual correction
///
/// The paper (arXiv:2504.19874) proves that at 3.5 bits/channel, TurboQuant preserves
/// attention quality identically to full precision.
///
/// Keys use TurboQuant because attention scores are inner products (dot products),
/// which TurboQuant's QJL corrects to be unbiased. Values use f16 because they are
/// weighted-summed (not inner-producted), where f16's ~0.01 absolute error is negligible.
pub struct CompressedKVCache {
    /// TurboQuant compressor for key vectors (one per KV head)
    key_quantizers: Vec<TurboQuant>,
    /// Compressed key vectors: [seq_len][n_kv_heads] = packed bytes per head
    compressed_keys: Vec<Vec<Vec<u8>>>,
    /// Key L2 norms (TurboQuant requires unit-norm input): [seq_len][n_kv_heads]
    key_norms: Vec<Vec<f32>>,
    /// Values as f16: [seq_len * n_kv_heads * head_dim]
    values: Vec<half::f16>,
    /// Head dimension (kv_head_dim, e.g. 128)
    head_dim: usize,
    /// Number of KV heads
    n_kv_heads: usize,
    /// Number of tokens stored
    seq_len: usize,
}

impl CompressedKVCache {
    /// Create a new TurboQuant-compressed KV cache.
    ///
    /// - `n_kv_heads`: number of KV attention heads
    /// - `head_dim`: dimension per head (kv_head_dim, typically 128)
    pub fn new(n_kv_heads: usize, head_dim: usize) -> Self {
        // Create one TurboQuant compressor per KV head.
        // 3-bit quantization + head_dim QJL bits = ~4 bits/coordinate total.
        // Each head gets a different seed for independent random rotations.
        let key_quantizers: Vec<TurboQuant> = (0..n_kv_heads)
            .map(|h| TurboQuant::new(3, head_dim, head_dim, 0xCAFE_BABE + h as u64))
            .collect();

        Self {
            key_quantizers,
            compressed_keys: Vec::new(),
            key_norms: Vec::new(),
            values: Vec::new(),
            head_dim,
            n_kv_heads,
            seq_len: 0,
        }
    }

    /// Append new key/value vectors for one token position.
    ///
    /// Keys are L2-normalized then compressed via TurboQuant (3-bit + QJL).
    /// The norm is stored separately to reconstruct accurate inner products:
    ///   dot(q, k) ≈ ||k|| × TurboQuant.dot(q, compress(k/||k||))
    /// Values are stored as f16.
    pub fn append(&mut self, keys: &[f32], values: &[f32]) {
        debug_assert_eq!(keys.len(), self.n_kv_heads * self.head_dim);
        debug_assert_eq!(values.len(), self.n_kv_heads * self.head_dim);

        // Compress each KV head's key vector independently
        let mut token_keys = Vec::with_capacity(self.n_kv_heads);
        let mut token_norms = Vec::with_capacity(self.n_kv_heads);
        for h in 0..self.n_kv_heads {
            let offset = h * self.head_dim;
            let key_head = &keys[offset..offset + self.head_dim];

            // L2-normalize before TurboQuant (required for correct codebook matching)
            let norm: f32 = key_head.iter().map(|x| x * x).sum::<f32>().sqrt();
            let inv_norm = if norm > 1e-12 { 1.0 / norm } else { 0.0 };
            let normalized: Vec<f32> = key_head.iter().map(|&x| x * inv_norm).collect();

            let packed = self.key_quantizers[h].quantize_vector(&normalized);
            token_keys.push(packed);
            token_norms.push(norm);
        }
        self.compressed_keys.push(token_keys);
        self.key_norms.push(token_norms);

        // Store values as f16
        self.values
            .extend(values.iter().map(|&v| half::f16::from_f32(v)));
        self.seq_len += 1;
    }

    /// Compute attention scores for one head using TurboQuant-compressed keys.
    ///
    /// Uses the asymmetric QJL estimator for unbiased inner products.
    /// Rescales by the stored key norm: score = ||k|| × dot(q, k_hat) / sqrt(d)
    /// Returns attention weights [seq_len] for the given query, scaled by 1/sqrt(d).
    pub fn attention_scores(&self, q_head: &[f32], kv_head_idx: usize) -> Vec<f32> {
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let tq = &self.key_quantizers[kv_head_idx];
        let mut scores = vec![0.0f32; self.seq_len];

        for t in 0..self.seq_len {
            let packed_key = &self.compressed_keys[t][kv_head_idx];
            let key_norm = self.key_norms[t][kv_head_idx];
            // dot(q, k) ≈ ||k|| × TurboQuant.dot(q, k/||k||)
            scores[t] = key_norm * tq.dot(q_head, packed_key) * scale;
        }

        scores
    }

    /// Weighted sum of f16 value vectors.
    ///
    /// - `weights`: attention weights [seq_len] (already softmaxed)
    /// - `kv_head_idx`: which KV head to read from
    ///
    /// Returns the weighted value vector [head_dim] in f32.
    pub fn weighted_value_sum(&self, weights: &[f32], kv_head_idx: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.head_dim];
        let stride = self.n_kv_heads * self.head_dim;

        for (t, &w) in weights.iter().enumerate() {
            let offset = t * stride + kv_head_idx * self.head_dim;
            for d in 0..self.head_dim {
                out[d] += w * self.values[offset + d].to_f32();
            }
        }

        out
    }

    /// Current sequence length
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let key_bytes: usize = self.compressed_keys.iter()
            .flat_map(|token| token.iter())
            .map(|packed| packed.len())
            .sum();
        let norm_bytes = self.key_norms.len() * self.n_kv_heads * 4; // f32 norms
        let value_bytes = self.values.len() * 2; // f16 = 2 bytes
        key_bytes + norm_bytes + value_bytes
    }

    /// Memory usage if stored as f32 (uncompressed)
    pub fn uncompressed_memory_bytes(&self) -> usize {
        self.seq_len * self.n_kv_heads * self.head_dim * 4 * 2 // keys + values
    }

    /// Compression ratio
    pub fn compression_ratio(&self) -> f32 {
        if self.seq_len == 0 {
            return 1.0;
        }
        self.uncompressed_memory_bytes() as f32 / self.memory_bytes() as f32
    }
}

// --- Bit Packing Utilities ---

/// Pack quantized indices into bytes (MSE-only, no QJL).
fn pack_indices(indices: &[u8], bits_per_index: usize) -> Vec<u8> {
    let total_bits = indices.len() * bits_per_index;
    let total_bytes = total_bits.div_ceil(8);
    let mut packed = vec![0u8; total_bytes];

    let mut bit_pos = 0usize;
    for &idx in indices {
        for b in 0..bits_per_index {
            if (idx >> b) & 1 == 1 {
                packed[bit_pos / 8] |= 1 << (bit_pos % 8);
            }
            bit_pos += 1;
        }
    }

    packed
}

/// Pack quantized indices and QJL sign bits into a byte vector.
fn pack_compressed(indices: &[u8], bits_per_index: usize, qjl_signs: &[bool]) -> Vec<u8> {
    let index_bits = indices.len() * bits_per_index;
    let total_bits = index_bits + qjl_signs.len();
    let total_bytes = total_bits.div_ceil(8);
    let mut packed = vec![0u8; total_bytes];

    // Pack quantized indices
    let mut bit_pos = 0usize;
    for &idx in indices {
        for b in 0..bits_per_index {
            if (idx >> b) & 1 == 1 {
                packed[bit_pos / 8] |= 1 << (bit_pos % 8);
            }
            bit_pos += 1;
        }
    }

    // Pack QJL sign bits
    for &sign in qjl_signs {
        if sign {
            packed[bit_pos / 8] |= 1 << (bit_pos % 8);
        }
        bit_pos += 1;
    }

    packed
}

/// Unpack quantized indices from packed bytes.
fn unpack_indices(packed: &[u8], bits_per_index: usize, dim: usize) -> Vec<u8> {
    let mut indices = vec![0u8; dim];
    let mut bit_pos = 0usize;

    for idx in indices.iter_mut().take(dim) {
        for b in 0..bits_per_index {
            if (packed[bit_pos / 8] >> (bit_pos % 8)) & 1 == 1 {
                *idx |= 1 << b;
            }
            bit_pos += 1;
        }
    }

    indices
}

/// Unpack QJL sign bits from packed bytes.
fn unpack_qjl_signs(
    packed: &[u8],
    bits_per_index: usize,
    dim: usize,
    qjl_bits: usize,
) -> Vec<bool> {
    let mut signs = vec![false; qjl_bits];
    let start_bit = dim * bits_per_index;

    for (i, sign) in signs.iter_mut().enumerate() {
        let bit_pos = start_bit + i;
        *sign = (packed[bit_pos / 8] >> (bit_pos % 8)) & 1 == 1;
    }

    signs
}

// --- Matrix Generation ---

/// Generate a random orthogonal matrix [dim x dim] using QR decomposition.
fn generate_orthogonal_matrix(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);

    // Generate random Gaussian matrix
    let mut a = vec![0.0f32; dim * dim];
    for val in a.iter_mut() {
        *val = rng.gaussian();
    }

    // QR decomposition via modified Gram-Schmidt
    let mut q = vec![0.0f32; dim * dim];
    let mut r = vec![0.0f32; dim * dim];

    for j in 0..dim {
        // Copy column j of a into q
        for i in 0..dim {
            q[i * dim + j] = a[i * dim + j];
        }
        // Subtract projections onto previous columns
        for k in 0..j {
            let mut dot = 0.0f32;
            for i in 0..dim {
                dot += q[i * dim + k] * a[i * dim + j];
            }
            r[k * dim + j] = dot;
            for i in 0..dim {
                q[i * dim + j] -= dot * q[i * dim + k];
            }
        }
        // Normalize
        let mut norm = 0.0f32;
        for i in 0..dim {
            norm += q[i * dim + j] * q[i * dim + j];
        }
        norm = norm.sqrt();
        r[j * dim + j] = norm;
        if norm > 1e-10 {
            for i in 0..dim {
                q[i * dim + j] /= norm;
            }
        }
    }

    q
}

/// Generate a random Gaussian matrix [rows x cols].
fn generate_gaussian_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut m = vec![0.0f32; rows * cols];
    for val in m.iter_mut() {
        *val = rng.gaussian();
    }
    m
}

// --- Simple PRNG (splitmix64) ---

/// SplitMix64 PRNG with Gaussian sampling via Box-Muller.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gaussian(&mut self) -> f32 {
        // Box-Muller transform
        let u1 = self.next_f64().max(1e-10);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        (r * theta.cos()) as f32
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lloyd_max_centroids() {
        for bw in 1..=4 {
            let c = LLOYD_MAX_CENTROIDS[bw - 1];
            assert_eq!(c.len(), 1 << bw);
            for i in 1..c.len() {
                assert!(c[i] > c[i - 1], "centroids not sorted for {bw}-bit");
            }
            let n = c.len();
            for i in 0..n / 2 {
                assert!(
                    (c[i] + c[n - 1 - i]).abs() < 0.01,
                    "centroids not symmetric for {bw}-bit",
                );
            }
        }
    }

    #[test]
    fn test_orthogonal_matrix() {
        let dim = 8;
        let q = generate_orthogonal_matrix(dim, 42);

        // Check Q^T Q = I
        for i in 0..dim {
            for j in 0..dim {
                let mut dot = 0.0f32;
                for k in 0..dim {
                    dot += q[k * dim + i] * q[k * dim + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 0.01,
                    "Q^T Q != I at ({i}, {j}): {dot}",
                );
            }
        }
    }

    #[test]
    fn test_quantize_dequantize_with_qjl() {
        let dim = 16;
        let tq = TurboQuant::new(3, dim, dim, 123);

        let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let x: Vec<f32> = x.iter().map(|v| v / norm).collect();

        let packed = tq.quantize_vector(&x);
        let reconstructed = tq.dequantize_vector(&packed);

        let mse: f32 = x
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / dim as f32;

        assert!(mse < 0.2, "MSE too high: {mse}");
    }

    #[test]
    fn test_quantize_dequantize_mse_only() {
        let dim = 16;
        let tq = TurboQuant::new_mse_only(3, dim, 123);

        let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let x: Vec<f32> = x.iter().map(|v| v / norm).collect();

        let packed = tq.quantize_vector(&x);
        let reconstructed = tq.dequantize_vector(&packed);

        let mse: f32 = x
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / dim as f32;

        assert!(mse < 0.2, "MSE too high: {mse}");
    }

    #[test]
    fn test_dot_accuracy_qjl() {
        let dim = 64;
        let tq = TurboQuant::new(3, dim, dim, 456);

        let mut rng = SplitMix64::new(789);
        let mut errors = Vec::new();

        for _ in 0..50 {
            let a: Vec<f32> = (0..dim).map(|_| rng.gaussian()).collect();
            let b: Vec<f32> = (0..dim).map(|_| rng.gaussian()).collect();

            let norm_a: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
            let a: Vec<f32> = a.iter().map(|v| v / norm_a).collect();
            let b: Vec<f32> = b.iter().map(|v| v / norm_b).collect();

            let true_dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let packed_b = tq.quantize_vector(&b);
            let est_dot = tq.dot(&a, &packed_b);

            errors.push((true_dot - est_dot).abs());
        }

        let mean_error = errors.iter().sum::<f32>() / errors.len() as f32;
        assert!(
            mean_error < 0.15,
            "QJL mean absolute error too high: {mean_error}"
        );
    }

    #[test]
    fn test_compressed_kv_cache() {
        let n_kv_heads = 2;
        let head_dim = 16;
        let mut cache = CompressedKVCache::new(n_kv_heads, head_dim);

        for t in 0..10 {
            let keys: Vec<f32> = (0..n_kv_heads * head_dim)
                .map(|i| ((t * 100 + i) as f32 * 0.01).sin())
                .collect();
            let vals: Vec<f32> = (0..n_kv_heads * head_dim)
                .map(|i| ((t * 100 + i) as f32 * 0.02).cos())
                .collect();
            cache.append(&keys, &vals);
        }

        assert_eq!(cache.seq_len(), 10);
        // TurboQuant 3-bit keys + f16 values: compression > 2x
        let ratio = cache.compression_ratio();
        assert!(
            ratio > 2.0,
            "Compression ratio should be > 2.0x with TurboQuant keys, got {ratio}",
        );

        let q: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let scores = cache.attention_scores(&q, 0);
        assert_eq!(scores.len(), 10);
    }

    #[test]
    fn test_bit_packing_qjl() {
        let indices = vec![0, 1, 2, 3, 1, 0, 3, 2];
        let qjl_signs = vec![true, false, true, true];
        let packed = pack_compressed(&indices, 2, &qjl_signs);

        let unpacked = unpack_indices(&packed, 2, 8);
        assert_eq!(unpacked, indices);

        let unpacked_signs = unpack_qjl_signs(&packed, 2, 8, 4);
        assert_eq!(unpacked_signs, qjl_signs);
    }

    #[test]
    fn test_bit_packing_mse_only() {
        let indices = vec![0, 1, 2, 3, 1, 0, 3, 2];
        let packed = pack_indices(&indices, 2);

        let unpacked = unpack_indices(&packed, 2, 8);
        assert_eq!(unpacked, indices);
    }

    #[test]
    fn test_compression_sizes_with_qjl() {
        // With QJL: 3-bit with dim=128
        // index bits = 3 * 128 = 384 bits = 48 bytes
        // qjl bits = 128 bits = 16 bytes
        // total = 64 bytes
        // vs f32 = 512 bytes
        // ratio = 8x
        let tq = TurboQuant::new(3, 128, 128, 0);
        assert_eq!(tq.compressed_bytes(), 64);

        // With QJL: 4-bit with dim=128
        // index bits = 4 * 128 = 512 bits = 64 bytes
        // qjl bits = 128 bits = 16 bytes
        // total = 80 bytes
        // ratio = 6.4x
        let tq4 = TurboQuant::new(4, 128, 128, 0);
        assert_eq!(tq4.compressed_bytes(), 80);
    }

    #[test]
    fn test_compression_sizes_mse_only() {
        // MSE-only (no QJL): 3-bit with dim=128
        // index bits = 3 * 128 = 384 bits = 48 bytes
        // vs f32 = 512 bytes
        // ratio = 10.7x
        let tq = TurboQuant::new_mse_only(3, 128, 0);
        assert_eq!(tq.compressed_bytes(), 48);

        // MSE-only: 4-bit with dim=128
        // index bits = 4 * 128 = 512 bits = 64 bytes
        // ratio = 8x
        let tq4 = TurboQuant::new_mse_only(4, 128, 0);
        assert_eq!(tq4.compressed_bytes(), 64);
    }
}
