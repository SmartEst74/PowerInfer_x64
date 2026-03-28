//! TurboQuant: Extreme KV cache compression for LLM inference
//!
//! Implements Google Research's TurboQuant algorithm (ICLR 2026) for compressing
//! key-value caches to 2-4 bits per coordinate with zero accuracy loss.
//!
//! Paper: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
//! arXiv:2504.19874
//!
//! Two-stage algorithm:
//! - Stage 1 (PolarQuant): Random rotation + Lloyd-Max scalar quantization
//! - Stage 2 (QJL): 1-bit sign correction for unbiased inner products

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
pub struct TurboQuant {
    /// Random orthogonal rotation matrix [dim x dim], row-major
    rotation: Vec<f32>,
    /// Dimensionality of vectors
    dim: usize,
    /// Bits per coordinate (2, 3, or 4)
    bits: usize,
    /// Lloyd-Max centroids for the selected bit-width
    centroids: Vec<f32>,
    /// QJL projection matrix [qjl_bits x dim], row-major, random Gaussian
    qjl_matrix: Vec<f32>,
    /// Number of QJL projection bits (1 bit per projection)
    qjl_bits: usize,
}

impl TurboQuant {
    /// Create a new TurboQuant compressor.
    ///
    /// - `bits`: quantization bits per coordinate (2, 3, or 4)
    /// - `dim`: vector dimensionality (typically head_dim, e.g. 128)
    /// - `qjl_bits`: number of QJL residual correction bits (typically = dim)
    /// - `seed`: RNG seed for reproducibility
    pub fn new(bits: usize, dim: usize, qjl_bits: usize, seed: u64) -> Self {
        assert!((2..=4).contains(&bits), "bits must be 2, 3, or 4");
        assert!(dim > 0, "dim must be > 0");

        let centroids = LLOYD_MAX_CENTROIDS[bits - 1].to_vec();

        // Generate random orthogonal matrix via QR decomposition
        let rotation = generate_orthogonal_matrix(dim, seed);

        // Generate QJL projection matrix (random Gaussian)
        let qjl_matrix = generate_gaussian_matrix(qjl_bits, dim, seed ^ 0xDEAD_BEEF);

        Self {
            rotation,
            dim,
            bits,
            centroids,
            qjl_matrix,
            qjl_bits,
        }
    }

    /// Number of bytes needed to store a compressed vector
    pub fn compressed_bytes(&self) -> usize {
        // Quantized indices: bits * dim bits, packed
        let index_bytes = (self.bits * self.dim).div_ceil(8);
        // QJL sign bits: qjl_bits bits, packed
        let qjl_bytes = self.qjl_bits.div_ceil(8);
        index_bytes + qjl_bytes
    }

    /// Quantize a single vector to packed bytes.
    ///
    /// Input: `x` [dim] — the vector to compress (assumed unit-norm)
    /// Output: packed bytes containing quantized indices + QJL sign bits
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

    /// Compute attention score between a full-precision query and a compressed key
    /// using the asymmetric TurboQuant estimator.
    ///
    /// This is the core operation — accurate attention scores from compressed keys.
    pub fn asymmetric_dot(&self, q: &[f32], packed_key: &[u8]) -> f32 {
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
        // Both q and k are unit vectors; after rotation they have variance 1/d.
        // The centroids are in N(0,1) space, so we need to scale the dot product.
        let mut mse_dot = 0.0f32;
        for i in 0..self.dim {
            let k_deq = self.centroids[indices[i] as usize];
            mse_dot += q_rot[i] * k_deq * inv_scale;
        }

        // Stage 2: QJL correction
        // Estimate residual norm from quantization distortion
        let distortion = self.expected_distortion();
        // In scaled space, variance is ~1, so residual variance ≈ distortion
        // Residual norm ≈ sqrt(distortion * dim) * inv_scale ≈ sqrt(distortion / dim)
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

// --- Compressed KV Cache ---

/// Compressed KV cache layer using TurboQuant.
///
/// Stores keys and values in compressed format, enabling much longer contexts
/// within the same memory budget.
pub struct CompressedKVCache {
    /// Compressed key vectors [seq_len * packed_bytes_per_key]
    keys_packed: Vec<u8>,
    /// Compressed value vectors [seq_len * packed_bytes_per_val]
    vals_packed: Vec<u8>,
    /// Bytes per compressed key vector
    key_packed_bytes: usize,
    /// Bytes per compressed value vector
    val_packed_bytes: usize,
    /// Number of head dimensions
    head_dim: usize,
    /// Number of KV heads
    n_kv_heads: usize,
    /// Number of tokens stored
    seq_len: usize,
    /// Key compressor
    key_quant: TurboQuant,
    /// Value compressor (may use higher bits for accuracy)
    val_quant: TurboQuant,
}

impl CompressedKVCache {
    /// Create a new compressed KV cache.
    ///
    /// - `n_kv_heads`: number of key/value attention heads
    /// - `head_dim`: dimension per head
    /// - `key_bits`: quantization bits for keys (2-4)
    /// - `val_bits`: quantization bits for values (2-4)
    /// - `seed`: RNG seed
    pub fn new(
        n_kv_heads: usize,
        head_dim: usize,
        key_bits: usize,
        val_bits: usize,
        seed: u64,
    ) -> Self {
        let key_quant = TurboQuant::new(key_bits, head_dim, head_dim, seed);
        let val_quant = TurboQuant::new(val_bits, head_dim, head_dim, seed ^ 0xCAFE);

        let key_packed_bytes = key_quant.compressed_bytes();
        let val_packed_bytes = val_quant.compressed_bytes();

        Self {
            keys_packed: Vec::new(),
            vals_packed: Vec::new(),
            key_packed_bytes,
            val_packed_bytes,
            head_dim,
            n_kv_heads,
            seq_len: 0,
            key_quant,
            val_quant,
        }
    }

    /// Append new key/value vectors for one token position.
    ///
    /// - `keys`: [n_kv_heads * head_dim] key vectors
    /// - `values`: [n_kv_heads * head_dim] value vectors
    pub fn append(&mut self, keys: &[f32], values: &[f32]) {
        debug_assert_eq!(keys.len(), self.n_kv_heads * self.head_dim);
        debug_assert_eq!(values.len(), self.n_kv_heads * self.head_dim);

        for h in 0..self.n_kv_heads {
            let h_offset = h * self.head_dim;
            let key_head = &keys[h_offset..h_offset + self.head_dim];
            let val_head = &values[h_offset..h_offset + self.head_dim];

            let packed_key = self.key_quant.quantize_vector(key_head);
            let packed_val = self.val_quant.quantize_vector(val_head);

            self.keys_packed.extend_from_slice(&packed_key);
            self.vals_packed.extend_from_slice(&packed_val);
        }

        self.seq_len += 1;
    }

    /// Compute attention scores for one head using compressed keys.
    ///
    /// Returns attention weights [seq_len] for the given query.
    pub fn attention_scores(&self, q_head: &[f32], kv_head_idx: usize) -> Vec<f32> {
        let mut scores = vec![0.0f32; self.seq_len];

        for t in 0..self.seq_len {
            let key_offset = (t * self.n_kv_heads + kv_head_idx) * self.key_packed_bytes;
            let packed_key = &self.keys_packed[key_offset..key_offset + self.key_packed_bytes];
            scores[t] = self.key_quant.asymmetric_dot(q_head, packed_key);
        }

        // Scale by 1/sqrt(d)
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        for s in scores.iter_mut() {
            *s *= scale;
        }

        scores
    }

    /// Weighted sum of compressed value vectors.
    ///
    /// - `weights`: attention weights [seq_len] (already softmaxed)
    /// - `kv_head_idx`: which KV head to read from
    ///
    /// Returns the weighted value vector [head_dim].
    pub fn weighted_value_sum(&self, weights: &[f32], kv_head_idx: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.head_dim];

        for (t, &weight) in weights.iter().enumerate() {
            let val_offset = (t * self.n_kv_heads + kv_head_idx) * self.val_packed_bytes;
            let packed_val = &self.vals_packed[val_offset..val_offset + self.val_packed_bytes];
            let deq_val = self.val_quant.dequantize_vector(packed_val);

            for (o, v) in out.iter_mut().zip(deq_val.iter()) {
                *o += weight * v;
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
        self.keys_packed.len() + self.vals_packed.len()
    }

    /// Memory usage if stored as f32 (uncompressed)
    pub fn uncompressed_memory_bytes(&self) -> usize {
        self.seq_len * self.n_kv_heads * self.head_dim * 4 * 2 // keys + values
    }

    /// Compression ratio
    pub fn compression_ratio(&self) -> f32 {
        if self.keys_packed.is_empty() {
            return 1.0;
        }
        self.uncompressed_memory_bytes() as f32 / self.memory_bytes() as f32
    }
}

// --- Bit Packing Utilities ---

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
        // Verify centroids are ordered and symmetric
        for bw in 1..=4 {
            let c = LLOYD_MAX_CENTROIDS[bw - 1];
            assert_eq!(c.len(), 1 << bw);
            // Should be sorted
            for i in 1..c.len() {
                assert!(c[i] > c[i - 1], "centroids not sorted for {bw}-bit");
            }
            // Should be approximately symmetric
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
    fn test_quantize_dequantize() {
        let dim = 16;
        let tq = TurboQuant::new(3, dim, dim, 123);

        // Random unit vector
        let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let x: Vec<f32> = x.iter().map(|v| v / norm).collect();

        let packed = tq.quantize_vector(&x);
        let reconstructed = tq.dequantize_vector(&packed);

        // Per-vector reconstruction won't be perfect, but should be reasonable
        let mse: f32 = x
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / dim as f32;

        // 3-bit MSE should be < 0.2 for unit vectors
        assert!(mse < 0.2, "MSE too high: {mse}");
    }

    #[test]
    fn test_asymmetric_dot_accuracy() {
        let dim = 64;
        let tq = TurboQuant::new(3, dim, dim, 456);

        // Generate random unit vectors and test inner product accuracy
        let mut rng = SplitMix64::new(789);
        let mut errors = Vec::new();

        for _ in 0..50 {
            let a: Vec<f32> = (0..dim).map(|_| rng.gaussian()).collect();
            let b: Vec<f32> = (0..dim).map(|_| rng.gaussian()).collect();

            // Normalize
            let norm_a: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
            let a: Vec<f32> = a.iter().map(|v| v / norm_a).collect();
            let b: Vec<f32> = b.iter().map(|v| v / norm_b).collect();

            // True dot product
            let true_dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

            // Compressed dot product
            let packed_b = tq.quantize_vector(&b);
            let est_dot = tq.asymmetric_dot(&a, &packed_b);

            errors.push((true_dot - est_dot).abs());
        }

        let mean_error = errors.iter().sum::<f32>() / errors.len() as f32;
        // 3-bit asymmetric dot should have mean error < 0.15
        assert!(
            mean_error < 0.15,
            "Mean absolute error too high: {mean_error}"
        );
    }

    #[test]
    fn test_compressed_kv_cache() {
        let n_kv_heads = 2;
        let head_dim = 16;
        let mut cache = CompressedKVCache::new(n_kv_heads, head_dim, 3, 3, 999);

        // Append some vectors
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
        assert!(
            cache.compression_ratio() > 3.0,
            "Compression ratio should be > 3x"
        );

        // Test attention scores
        let q: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let scores = cache.attention_scores(&q, 0);
        assert_eq!(scores.len(), 10);
    }

    #[test]
    fn test_bit_packing() {
        let indices = vec![0, 1, 2, 3, 1, 0, 3, 2];
        let qjl_signs = vec![true, false, true, true];
        let packed = pack_compressed(&indices, 2, &qjl_signs);

        let unpacked = unpack_indices(&packed, 2, 8);
        assert_eq!(unpacked, indices);

        let unpacked_signs = unpack_qjl_signs(&packed, 2, 8, 4);
        assert_eq!(unpacked_signs, qjl_signs);
    }

    #[test]
    fn test_compression_sizes() {
        // At 3-bit with dim=128:
        // index bits = 3 * 128 = 384 bits = 48 bytes
        // qjl bits = 128 bits = 16 bytes
        // total = 64 bytes
        // vs f32 = 512 bytes
        // ratio = 8x
        let tq = TurboQuant::new(3, 128, 128, 0);
        assert_eq!(tq.compressed_bytes(), 64);

        // At 4-bit with dim=128:
        // index bits = 4 * 128 = 512 bits = 64 bytes
        // qjl bits = 128 bits = 16 bytes
        // total = 80 bytes
        // vs f32 = 512 bytes
        // ratio = 6.4x
        let tq4 = TurboQuant::new(4, 128, 128, 0);
        assert_eq!(tq4.compressed_bytes(), 80);
    }
}
