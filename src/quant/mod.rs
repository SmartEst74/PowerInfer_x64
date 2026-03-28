//! Quantization utilities and CPU dequantization kernels
//!
//! Implements GGML-compatible dequantization for common quantization types.
//! Reference: llama.cpp quantize.cpp and ggml-quants.c

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

    /// Number of bytes per block for this quantization type
    pub fn block_size_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18, // 2 (scale) + 16 (weights)
            Self::Q4_1 => 20, // 2 (scale) + 2 (min) + 16 (weights)
            Self::Q5_0 => 22, // 2 (scale) + 4 (qh) + 16 (weights)
            Self::Q5_1 => 24, // 2 (scale) + 2 (min) + 4 (qh) + 16 (weights)
            Self::Q8_0 => 34, // 2 (scale) + 32 (weights)
            Self::Q2_K => 84,
            Self::Q3_K_S | Self::Q3_K_M | Self::Q3_K_L => 110,
            Self::Q4_K_S | Self::Q4_K_M => 144,
            Self::Q5_K_S | Self::Q5_K_M => 176,
            Self::Q6_K => 210,
            _ => 0,
        }
    }

    /// Number of values per block
    pub fn values_per_block(&self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 => 32,
            Self::Q2_K => 256,
            Self::Q3_K_S | Self::Q3_K_M | Self::Q3_K_L => 256,
            Self::Q4_K_S | Self::Q4_K_M => 256,
            Self::Q5_K_S | Self::Q5_K_M => 256,
            Self::Q6_K => 256,
            _ => 32,
        }
    }
}

/// Dequantize an entire weight tensor to f32.
///
/// Returns a Vec<f32> with `rows * cols` elements in row-major order.
pub fn dequantize(
    data: &[u8],
    qtype: QuantizationType,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    let block_size = qtype.block_size_bytes();
    let vals_per_block = qtype.values_per_block();
    let blocks_per_row = cols / vals_per_block;
    let total_blocks = rows * blocks_per_row;
    let expected_bytes = total_blocks * block_size;

    if data.len() < expected_bytes {
        return Err(anyhow!(
            "Insufficient data for dequantization: need {} bytes, have {}",
            expected_bytes,
            data.len()
        ));
    }

    match qtype {
        QuantizationType::F32 => dequantize_f32(data, rows, cols),
        QuantizationType::F16 => dequantize_f16(data, rows, cols),
        QuantizationType::Q4_0 => dequantize_q4_0(data, rows, cols),
        QuantizationType::Q4_1 => dequantize_q4_1(data, rows, cols),
        QuantizationType::Q5_0 => dequantize_q5_0(data, rows, cols),
        QuantizationType::Q5_1 => dequantize_q5_1(data, rows, cols),
        QuantizationType::Q8_0 => dequantize_q8_0(data, rows, cols),
        _ => Err(anyhow!("Dequantization not yet implemented for {qtype:?}")),
    }
}

/// Dequantize F32 (identity, just reinterpret bytes)
fn dequantize_f32(data: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let n = rows * cols;
    if data.len() < n * 4 {
        return Err(anyhow!("Insufficient data for F32 dequantization"));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * 4;
        let bytes: [u8; 4] = data[offset..offset + 4].try_into().unwrap();
        out.push(f32::from_le_bytes(bytes));
    }
    Ok(out)
}

/// Dequantize F16 to f32
fn dequantize_f16(data: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let n = rows * cols;
    if data.len() < n * 2 {
        return Err(anyhow!("Insufficient data for F16 dequantization"));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * 2;
        let bytes: [u8; 2] = data[offset..offset + 2].try_into().unwrap();
        let h = f16::from_le_bytes(bytes);
        out.push(h.to_f32());
    }
    Ok(out)
}

// --- Q4_0 ---
// Block: 2 bytes (f16 scale) + 16 bytes (32 x 4-bit weights)
// Each byte holds 2 weights: low nibble and high nibble
// Weight = (nibble - 8) * scale
fn dequantize_q4_0(data: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let vals_per_block = 32usize;
    let blocks_per_row = cols / vals_per_block;
    let mut out = Vec::with_capacity(rows * cols);

    for _row in 0..rows {
        for _block in 0..blocks_per_row {
            let block_offset = (out.len() / vals_per_block) * 18;
            // Scale: f16 at bytes 0-1
            let scale_bytes: [u8; 2] = data[block_offset..block_offset + 2].try_into().unwrap();
            let scale = f16::from_le_bytes(scale_bytes).to_f32();

            // Weights: 16 bytes starting at byte 2
            for i in 0..16 {
                let byte = data[block_offset + 2 + i];
                let lo = (byte & 0x0F) as i32 - 8;
                let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                out.push(lo as f32 * scale);
                out.push(hi as f32 * scale);
            }
        }
    }

    Ok(out)
}

// --- Q4_1 ---
// Block: 2 bytes (f16 d) + 2 bytes (f16 m) + 16 bytes (32 x 4-bit weights)
// Weight = nibble * d + m
fn dequantize_q4_1(data: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let vals_per_block = 32usize;
    let blocks_per_row = cols / vals_per_block;
    let mut out = Vec::with_capacity(rows * cols);

    for _row in 0..rows {
        for _block in 0..blocks_per_row {
            let block_offset = (out.len() / vals_per_block) * 20;
            let d_bytes: [u8; 2] = data[block_offset..block_offset + 2].try_into().unwrap();
            let d = f16::from_le_bytes(d_bytes).to_f32();
            let m_bytes: [u8; 2] = data[block_offset + 2..block_offset + 4].try_into().unwrap();
            let m = f16::from_le_bytes(m_bytes).to_f32();

            for i in 0..16 {
                let byte = data[block_offset + 4 + i];
                let lo = (byte & 0x0F) as f32;
                let hi = ((byte >> 4) & 0x0F) as f32;
                out.push(lo * d + m);
                out.push(hi * d + m);
            }
        }
    }

    Ok(out)
}

// --- Q5_0 ---
// Block: 2 bytes (f16 scale) + 4 bytes (high bits for 32 weights) + 16 bytes (low nibbles)
// Each weight is 5-bit: 4 low bits from nibble + 1 high bit
// Weight = (q5 - 16) * scale
fn dequantize_q5_0(data: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let vals_per_block = 32usize;
    let blocks_per_row = cols / vals_per_block;
    let mut out = Vec::with_capacity(rows * cols);

    for _row in 0..rows {
        for _block in 0..blocks_per_row {
            let block_offset = (out.len() / vals_per_block) * 22;
            let scale_bytes: [u8; 2] = data[block_offset..block_offset + 2].try_into().unwrap();
            let scale = f16::from_le_bytes(scale_bytes).to_f32();

            // High bits: 4 bytes, 1 bit per weight
            let qh = [
                data[block_offset + 2],
                data[block_offset + 3],
                data[block_offset + 4],
                data[block_offset + 5],
            ];

            // Low nibbles: 16 bytes
            for i in 0..16 {
                let byte = data[block_offset + 6 + i];
                let lo = byte & 0x0F;
                let hi = (byte >> 4) & 0x0F;

                // Reconstruct 5-bit values
                let q0 = lo | (((qh[i / 8] >> (i % 8)) & 1) << 4);
                let q1 = hi | (((qh[(i + 16) / 8] >> ((i + 16) % 8)) & 1) << 4);

                out.push((q0 as i32 - 16) as f32 * scale);
                out.push((q1 as i32 - 16) as f32 * scale);
            }
        }
    }

    Ok(out)
}

// --- Q5_1 ---
// Block: 2 bytes (f16 d) + 2 bytes (f16 m) + 4 bytes (high bits) + 16 bytes (low nibbles)
// Weight = q5 * d + m
fn dequantize_q5_1(data: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let vals_per_block = 32usize;
    let blocks_per_row = cols / vals_per_block;
    let mut out = Vec::with_capacity(rows * cols);

    for _row in 0..rows {
        for _block in 0..blocks_per_row {
            let block_offset = (out.len() / vals_per_block) * 24;
            let d_bytes: [u8; 2] = data[block_offset..block_offset + 2].try_into().unwrap();
            let d = f16::from_le_bytes(d_bytes).to_f32();
            let m_bytes: [u8; 2] = data[block_offset + 2..block_offset + 4].try_into().unwrap();
            let m = f16::from_le_bytes(m_bytes).to_f32();

            let qh = [
                data[block_offset + 4],
                data[block_offset + 5],
                data[block_offset + 6],
                data[block_offset + 7],
            ];

            for i in 0..16 {
                let byte = data[block_offset + 8 + i];
                let lo = byte & 0x0F;
                let hi = (byte >> 4) & 0x0F;

                let q0 = lo | (((qh[i / 8] >> (i % 8)) & 1) << 4);
                let q1 = hi | (((qh[(i + 16) / 8] >> ((i + 16) % 8)) & 1) << 4);

                out.push(q0 as f32 * d + m);
                out.push(q1 as f32 * d + m);
            }
        }
    }

    Ok(out)
}

// --- Q8_0 ---
// Block: 2 bytes (f16 scale) + 32 bytes (32 x 8-bit signed weights)
// Weight = int8 * scale
fn dequantize_q8_0(data: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let vals_per_block = 32usize;
    let blocks_per_row = cols / vals_per_block;
    let mut out = Vec::with_capacity(rows * cols);

    for _row in 0..rows {
        for _block in 0..blocks_per_row {
            let block_offset = (out.len() / vals_per_block) * 34;
            let scale_bytes: [u8; 2] = data[block_offset..block_offset + 2].try_into().unwrap();
            let scale = f16::from_le_bytes(scale_bytes).to_f32();

            for i in 0..32 {
                let q = data[block_offset + 2 + i] as i8 as f32;
                out.push(q * scale);
            }
        }
    }

    Ok(out)
}

/// Matrix-vector multiplication: y = x @ w
/// where x is [n_in], w is [n_out, n_in] (row-major), y is [n_out]
pub fn matvec_f32(y: &mut [f32], x: &[f32], w: &[f32], n_out: usize, n_in: usize) {
    for (i, yi) in y.iter_mut().enumerate().take(n_out) {
        let mut sum = 0.0f32;
        let row_offset = i * n_in;
        for j in 0..n_in {
            sum += x[j] * w[row_offset + j];
        }
        *yi = sum;
    }
}

/// Quantized matrix-vector multiplication: y = x @ w
/// where x is [n_in] (f32), w_data is quantized [n_out, n_in], y is [n_out] (f32)
pub fn matvec_quantized(
    y: &mut [f32],
    x: &[f32],
    w_data: &[u8],
    w_type: QuantizationType,
    n_out: usize,
    n_in: usize,
) -> Result<()> {
    let block_size = w_type.block_size_bytes();
    let vals_per_block = w_type.values_per_block();
    let blocks_per_row = n_in / vals_per_block;
    let bytes_per_row = blocks_per_row * block_size;

    for (i, yi) in y.iter_mut().enumerate().take(n_out) {
        let row_start = i * bytes_per_row;
        let row_data = &w_data[row_start..row_start + bytes_per_row];

        let dequant_row = dequantize(row_data, w_type, 1, n_in)?;
        let mut sum = 0.0f32;
        for j in 0..n_in {
            sum += x[j] * dequant_row[j];
        }
        *yi = sum;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequant_q4_0_roundtrip() {
        // Create a minimal Q4_0 block: scale=1.0 (f16), 16 bytes of weight data
        let scale = f16::from_f32(1.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale.to_le_bytes());
        // 16 bytes: all zeros → 32 weights of (0-8)=-8 → -8.0 each
        block.extend_from_slice(&[0u8; 16]);

        let out = dequantize_q4_0(&block, 1, 32).unwrap();
        assert_eq!(out.len(), 32);
        for v in &out {
            assert!((v - (-8.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dequant_q8_0_roundtrip() {
        let scale = f16::from_f32(0.5);
        let mut block = Vec::new();
        block.extend_from_slice(&scale.to_le_bytes());
        // 32 bytes of value 1 (as i8)
        block.extend_from_slice(&[1u8; 32]);

        let out = dequantize_q8_0(&block, 1, 32).unwrap();
        assert_eq!(out.len(), 32);
        for v in &out {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_matvec_f32() {
        // y = [1, 2] @ [[3, 4], [5, 6]] = [1*3+2*4, 1*5+2*6] = [11, 17]
        let x = [1.0f32, 2.0];
        let w = [3.0f32, 4.0, 5.0, 6.0];
        let mut y = [0.0f32; 2];
        matvec_f32(&mut y, &x, &w, 2, 2);
        assert!((y[0] - 11.0).abs() < 1e-6);
        assert!((y[1] - 17.0).abs() < 1e-6);
    }

    #[test]
    fn test_f32_dequantize() {
        let data: Vec<u8> = [0.0f32, 1.0, 2.0, 3.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let out = dequantize_f32(&data, 1, 4).unwrap();
        assert_eq!(out, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_q4_1_dequantize() {
        let d = f16::from_f32(1.0);
        let m = f16::from_f32(-8.0);
        let mut block = Vec::new();
        block.extend_from_slice(&d.to_le_bytes());
        block.extend_from_slice(&m.to_le_bytes());
        block.extend_from_slice(&[0u8; 16]);

        let out = dequantize_q4_1(&block, 1, 32).unwrap();
        assert_eq!(out.len(), 32);
        // nibble=0: 0*1.0 + (-8.0) = -8.0
        for v in &out {
            assert!((v - (-8.0)).abs() < 1e-6);
        }
    }
}
