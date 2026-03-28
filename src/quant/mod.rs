//! Quantization utilities

use anyhow::{anyhow, Result};
use half::f16;

/// Quantization type enum (mirrors GGML types)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum QuantizationType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2_K,
    Q3_K_S,
    Q3_K_M,
    Q3_K_L,
    Q4_K_S,
    Q4_K_M,
    Q5_K_S,
    Q5_K_M,
    Q6_K,
    IQ1_S,
    IQ1_M,
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,
    IQ3_XXS,
    IQ3_S,
    IQ3_M,
    IQ4_NL,
    IQ4_XS,
    F16,
    F32,
}

impl QuantizationType {
    pub fn from_ggml_type(ty: gguf_rs::GGMLType) -> Result<Self> {
        use gguf_rs::GGMLType as G;
        match ty {
            G::F32 => Ok(Self::F32),
            G::F16 => Ok(Self::F16),
            G::Q4_0 => Ok(Self::Q4_0),
            G::Q4_1 => Ok(Self::Q4_1),
            G::Q5_0 => Ok(Self::Q5_0),
            G::Q5_1 => Ok(Self::Q5_1),
            G::Q8_0 => Ok(Self::Q8_0),
            G::Q2_K => Ok(Self::Q2_K),
            G::Q3_K => Ok(Self::Q3_K_M),
            G::Q4_K => Ok(Self::Q4_K_M),
            G::Q5_K => Ok(Self::Q5_K_M),
            G::Q6_K => Ok(Self::Q6_K),
            G::IQ1_S => Ok(Self::IQ1_S),
            G::IQ1_M => Ok(Self::IQ1_M),
            G::IQ2_XXS => Ok(Self::IQ2_XXS),
            G::IQ2_XS => Ok(Self::IQ2_XS),
            G::IQ2_S => Ok(Self::IQ2_S),
            G::IQ3_XXS => Ok(Self::IQ3_XXS),
            G::IQ3_S => Ok(Self::IQ3_S),
            G::IQ4_NL => Ok(Self::IQ4_NL),
            G::IQ4_XS => Ok(Self::IQ4_XS),
            G::BF16 => Ok(Self::F16),
            other => Err(anyhow!("Unsupported GGML type: {other:?}")),
        }
    }
}

/// Dequantize a block of quantized data to f16
/// This is a placeholder - will be implemented with SIMD
pub fn dequantize_block(_data: &[u8], _qtype: QuantizationType, _out: &mut [f16], _offset: usize) {
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
    _weights: &[u8], // quantized
    _qtype: QuantizationType,
    _input: &[f16],
    _output: &mut [f16],
) {
    // Placeholder - real impl in GPU kernel or CPU fallback
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_dequant_q4_0() {
        // Will test against reference
    }
}
