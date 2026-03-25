//! Quantization utilities

use half::f16;

/// Quantization type enum (mirrored from gguf)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    Q4_0,
    Q4_K_M,
    Q5_K_M,
    Q8_0,
    F16,
    F32,
    // ... others
}

/// Dequantize a block of quantized data to f16
/// This is a placeholder - will be implemented with SIMD
pub fn dequantize_block(
    _data: &[u8],
    _qtype: QuantizationType,
    _out: &mut [f16],
    _offset: usize,
) {
    // Placeholder: will implement actual dequantization
    // This matches llama.cpp's dequant functions
}

/// Quantized matrix multiplication (SPARSE variant)
/// Computes: output = input @ weights.T, where only hot rows of weights are used
/// 
/// # Arguments
/// - `hot_indices`: indices of output neurons to compute (on GPU)
/// - `weights`: full weight matrix [n_out, n_in] in quantized form
/// - `input`: activation vector [n_in]
/// - `output`: output vector [n_out]; only hot entries written
pub fn sparse_matmul_hot(
    _hot_indices: &[u32],
    _weights: &[u8],  // quantized
    _qtype: QuantizationType,
    _input: &[f16],
    _output: &mut [f16],
) {
    // Placeholder - real impl in GPU kernel or CPU fallback
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dequant_q4_0() {
        // Will test against reference
    }
}
