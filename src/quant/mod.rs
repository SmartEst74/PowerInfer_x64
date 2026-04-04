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
            Self::Q6_K => 212,
            other => panic!("block_size_bytes not defined for {other:?}"),
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
            other => panic!("values_per_block not defined for {other:?}"),
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
        QuantizationType::Q4_K_S | QuantizationType::Q4_K_M => dequantize_q4_k(data, rows, cols),
        QuantizationType::Q5_K_S | QuantizationType::Q5_K_M => dequantize_q5_k(data, rows, cols),
        QuantizationType::Q6_K => dequantize_q6_k(data, rows, cols),
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

// --- K-quants (superblock format from llama.cpp) ---
// Reference: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c

/// Extract 6-bit scale and min values from Q4_K/Q5_K packed scales.
/// The scales array is 8 bytes, packing 8 scale values and 8 min values as 6-bit each.
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        let d = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (d, m)
    } else {
        let d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize Q4_K (also called Q4_K_M / Q4_K_S in GGUF).
///
/// Block: 144 bytes for 256 values.
/// Layout:
///   bytes 0-3:   d (f16), dmin (f16)
///   bytes 4-15:  scales (12 bytes, 6-bit packed scale+min per group of 32)
///   bytes 16-143: qs (128 bytes, 4-bit quantized values)
///
/// Dequant formula: value = d * sc * nibble - dmin * m
fn dequantize_q4_k(data: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let qk_k = 256usize;
    let block_size = 144usize;
    let blocks_per_row = cols / qk_k;
    let total_blocks = rows * blocks_per_row;
    let expected = total_blocks * block_size;

    if data.len() < expected {
        return Err(anyhow!(
            "Insufficient data for Q4_K dequant: need {expected}, have {}",
            data.len()
        ));
    }

    let actual_blocks = data.len() / block_size;
    let total_blocks = total_blocks.min(actual_blocks);
    let mut out = Vec::with_capacity(total_blocks * qk_k);
    let mut offset = 0usize;

    for _ in 0..total_blocks {
        let d = f16::from_le_bytes([data[offset], data[offset + 1]]).to_f32();
        let dmin = f16::from_le_bytes([data[offset + 2], data[offset + 3]]).to_f32();

        // 12 bytes of packed 6-bit scales/mins
        let scales = &data[offset + 4..offset + 16];

        // 128 bytes of 4-bit quantized values
        let qs = &data[offset + 16..offset + 144];

        let mut is = 0usize;
        let mut qi = 0usize;

        for _ in 0..4 {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            for j in 0..16 {
                out.push(d1 * (qs[qi + j] & 0xF) as f32 - m1);
            }
            for j in 0..16 {
                out.push(d1 * (qs[qi + j] >> 4) as f32 - m1);
            }
            is += 1;
            qi += 16;

            let (sc, m) = get_scale_min_k4(is, scales);
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;
            for j in 0..16 {
                out.push(d2 * (qs[qi + j] & 0xF) as f32 - m2);
            }
            for j in 0..16 {
                out.push(d2 * (qs[qi + j] >> 4) as f32 - m2);
            }
            is += 1;
            qi += 16;
        }

        offset += block_size;
    }

    Ok(out)
}

/// Dequantize Q5_K (also called Q5_K_M / Q5_K_S in GGUF).
///
/// Block: 176 bytes for 256 values.
/// Layout:
///   bytes 0-3:   d (f16), dmin (f16)
///   bytes 4-15:  scales (12 bytes, same as Q4_K)
///   bytes 16-47: qh (32 bytes, high bits for 5-bit values)
///   bytes 48-175: ql (128 bytes, low 4 bits)
fn dequantize_q5_k(data: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let qk_k = 256usize;
    let block_size = 176usize;
    let blocks_per_row = cols / qk_k;
    let total_blocks = rows * blocks_per_row;
    let expected = total_blocks * block_size;

    if data.len() < expected {
        return Err(anyhow!(
            "Insufficient data for Q5_K dequant: need {expected}, have {}",
            data.len()
        ));
    }

    let actual_blocks = data.len() / block_size;
    let total_blocks = total_blocks.min(actual_blocks);
    let mut out = Vec::with_capacity(total_blocks * qk_k);
    let mut offset = 0usize;

    for _ in 0..total_blocks {
        let d = f16::from_le_bytes([data[offset], data[offset + 1]]).to_f32();
        let dmin = f16::from_le_bytes([data[offset + 2], data[offset + 3]]).to_f32();

        let scales = &data[offset + 4..offset + 16];
        let qh = &data[offset + 16..offset + 48];
        let ql = &data[offset + 48..offset + 176];

        let mut is = 0usize;
        let mut qi = 0usize;
        let mut m1 = 1u8;
        let mut m2 = 2u8;

        for _ in 0..4 {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1f = dmin * m as f32;

            for j in 0..16 {
                let lo = ql[qi + j] & 0xF;
                let hi = if (qh[j] & m1) != 0 { 16 } else { 0 };
                out.push(d1 * (lo | hi) as f32 - m1f);
            }
            for j in 16..32 {
                let lo = ql[qi + j - 16] >> 4;
                let hi = if (qh[j] & m1) != 0 { 16 } else { 0 };
                out.push(d1 * (lo | hi) as f32 - m1f);
            }

            is += 1;
            qi += 16;

            let (sc, m) = get_scale_min_k4(is, scales);
            let d2 = d * sc as f32;
            let m2f = dmin * m as f32;

            for j in 0..16 {
                let lo = ql[qi + j] & 0xF;
                let hi = if (qh[j] & m2) != 0 { 16 } else { 0 };
                out.push(d2 * (lo | hi) as f32 - m2f);
            }
            for j in 16..32 {
                let lo = ql[qi + j - 16] >> 4;
                let hi = if (qh[j] & m2) != 0 { 16 } else { 0 };
                out.push(d2 * (lo | hi) as f32 - m2f);
            }

            is += 1;
            qi += 16;
            m1 <<= 2;
            m2 <<= 2;
        }

        offset += block_size;
    }

    Ok(out)
}

/// Dequantize Q6_K.
///
/// Block: 210 bytes for 256 values.
/// Layout:
///   bytes 0-1:   d (f16)
///   bytes 2-129: ql (128 bytes, low 4 bits, 2 values per byte)
///   bytes 130-193: qh (64 bytes, high 2 bits, 4 values per byte)
///   bytes 194-211: scales (18 bytes, signed i8, shifted by 32)
///
/// Dequant formula: value = d * scale * (ql + 16*qh - 32)
fn dequantize_q6_k(data: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let qk_k = 256usize;
    let block_size = 212usize; // d(2) + ql(128) + qh(64) + scales(18) = 212
    let blocks_per_row = cols / qk_k;
    let total_blocks = rows * blocks_per_row;
    let expected = total_blocks * block_size;

    if data.len() < expected {
        return Err(anyhow!(
            "Insufficient data for Q6_K dequant: need {expected}, have {}",
            data.len()
        ));
    }

    // Use the actual available data, rounded down to exact block boundaries
    let total_blocks = (data.len() / block_size).min(total_blocks);
    let mut out = Vec::with_capacity(total_blocks * qk_k);
    let mut offset = 0usize;

    for _ in 0..total_blocks {
        let d = f16::from_le_bytes([data[offset], data[offset + 1]]).to_f32();

        let ql = &data[offset + 2..offset + 130]; // 128 bytes
        let qh = &data[offset + 130..offset + 194]; // 64 bytes
        let scales = &data[offset + 194..offset + 212]; // 18 bytes, i8 shifted by 32

        let mut is = 0usize;

        for _super in 0..2 {
            for _group in 0..4 {
                let sc = (scales[is] as i32) - 32;
                let d1 = d * sc as f32;

                // Each group: 32 values
                // ql stores low 4 bits: 2 values per byte, 16 bytes per 32 values
                // qh stores high 2 bits: 4 values per byte, 8 bytes per 32 values
                let ql_offset = (is / 4) * 64 + (is % 4) * 16; // position in ql
                let qh_offset = (is / 4) * 32 + (is % 4) * 8; // position in qh

                for j in 0..32 {
                    let lo = if j < 16 {
                        ql[ql_offset + j] & 0xF
                    } else {
                        ql[ql_offset + j - 16] >> 4
                    };

                    let qh_byte = qh[qh_offset + j / 4];
                    let hi = ((qh_byte >> (2 * (j % 4))) & 3) as i32;

                    let q6 = lo as i32 + 16 * hi - 32;
                    out.push(d1 * q6 as f32);
                }

                is += 1;
            }
        }

        offset += block_size;
    }

    Ok(out)
}

/// Matrix-vector multiplication: y = x @ w
/// where x is [n_in], w is [n_out, n_in] (row-major), y is [n_out]
pub fn matvec_f32(y: &mut [f32], x: &[f32], w: &[f32], n_out: usize, n_in: usize) {
    let w_len = w.len();
    for (i, yi) in y.iter_mut().enumerate().take(n_out) {
        let mut sum = 0.0f32;
        let row_offset = i * n_in;
        if row_offset + n_in > w_len {
            break;
        }
        for j in 0..n_in {
            sum += x[j] * w[row_offset + j];
        }
        *yi = sum;
    }
}

/// Quantized matrix-vector multiplication: y = x @ w (row-major layout)
/// where x is [n_in] (f32), w_data is quantized [n_out, n_in] row-major, y is [n_out] (f32)
#[allow(dead_code)]
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

/// Compute y[j] = x · W[:,j] for all j, where W is [ne0=n_in, ne1=n_out] stored
/// in GGML column-major Q8_0 format.
///
/// This is the correct operation for GGML `ggml_mul_mat(W, x)` → y.
///
/// **GGML-compatible**: quantizes the f32 input vector to Q8_0 first, then
/// computes the dot product using integer accumulation (i8×i8→i32), matching
/// GGML's `ggml_vec_dot_q8_0_q8_0` exactly. This ensures identical numerical
/// results to llama.cpp, which is critical for MoE expert routing stability.
///
/// Uses SSE4.1 SIMD when available.
/// Parallelized across CPU cores using std::thread::scope for large matvecs.
///
/// Precondition: n_in must be divisible by 32 (Q8_0 block size).
pub fn matvec_col_major_q8_0(y: &mut [f32], x: &[f32], raw: &[u8], n_in: usize, n_out: usize) {
    debug_assert_eq!(x.len(), n_in);
    debug_assert_eq!(y.len(), n_out);
    debug_assert_eq!(n_in % 32, 0, "n_in must be divisible by 32 for Q8_0");

    let blocks_per_col = n_in / 32;
    let bytes_per_col = blocks_per_col * 34; // 2 scale bytes + 32 int8 bytes per block

    // Step 1: Quantize input f32 vector to Q8_0 blocks (matching GGML's quantize_row_q8_0)
    let x_q8 = quantize_f32_to_q8_0(x);

    // Parallelize large matvecs across available CPU cores.
    let n_threads = num_cpus::get().max(1);
    if n_threads > 1 && n_out >= 4096 {
        matvec_q8_0_parallel(y, &x_q8, raw, n_in, n_out, blocks_per_col, bytes_per_col, n_threads);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            unsafe { matvec_q8_0_sse(y, &x_q8, raw, n_in, n_out, blocks_per_col, bytes_per_col) };
            return;
        }
    }

    // Scalar fallback
    matvec_q8_0_scalar(y, &x_q8, raw, n_out, blocks_per_col, bytes_per_col);
}

/// Q8_0 block: 2 bytes scale (f16) + 32 bytes quantized values (i8)
struct Q8Block {
    scale: f32,
    qs: [i8; 32],
}

/// Quantize f32 vector to Q8_0 blocks, matching GGML's `quantize_row_q8_0` exactly.
fn quantize_f32_to_q8_0(x: &[f32]) -> Vec<Q8Block> {
    let n_blocks = x.len().div_ceil(32);
    let mut blocks = Vec::with_capacity(n_blocks);

    for b in 0..n_blocks {
        let base = b * 32;
        let end = (base + 32).min(x.len());

        // Find max absolute value in this block
        let mut amax = 0.0f32;
        for &v in &x[base..end] {
            let abs_v = v.abs();
            if abs_v > amax {
                amax = abs_v;
            }
        }

        // Compute scale (matching GGML: d = amax / 127, then store as f16)
        // CRITICAL: GGML uses 1.0/d (original f32) for quantization, NOT 1.0/f16_roundtrip(d)
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0 };
        let d_f16 = f16::from_f32(d);
        let d_f32 = d_f16.to_f32(); // Scale stored/used in dot product is f16 round-tripped

        let mut qs = [0i8; 32];
        for i in base..end {
            // NOTE: GGML uses roundf(), which rounds to nearest even.
            // Rust's f32::round() rounds away from zero for 0.5.
            // Use the GGML-compatible rounding: (val + 0.5).floor() for positive,
            // which is equivalent to roundf() in C for most values.
            let v = x[i] * id;
            qs[i - base] = v.round().clamp(-127.0, 127.0) as i8;
        }

        blocks.push(Q8Block { scale: d_f32, qs });
    }

    blocks
}

/// Parallel Q8_0 matvec: splits output rows across threads.
/// Uses GGML-compatible integer dot product (Q8_0 × Q8_0).
#[allow(clippy::too_many_arguments)]
fn matvec_q8_0_parallel(
    y: &mut [f32],
    x_q8: &[Q8Block],
    raw: &[u8],
    _n_in: usize,
    n_out: usize,
    blocks_per_col: usize,
    bytes_per_col: usize,
    n_threads: usize,
) {
    let chunk_size = n_out.div_ceil(n_threads);

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_threads);
        let mut y_rest = &mut *y;

        for chunk_idx in 0..n_threads {
            let start = chunk_idx * chunk_size;
            if start >= n_out {
                break;
            }
            let end = (start + chunk_size).min(n_out);
            let count = end - start;

            let (y_chunk, rest) = y_rest.split_at_mut(count);
            y_rest = rest;

            handles.push(s.spawn(move || {
                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("sse4.1") {
                        unsafe {
                            matvec_q8_0_sse_range(
                                y_chunk, x_q8, raw, start, count,
                                blocks_per_col, bytes_per_col,
                            );
                        }
                        return;
                    }
                }
                // Scalar fallback for this chunk
                for (local_j, yj) in y_chunk.iter_mut().enumerate() {
                    let j = start + local_j;
                    let col_off = j * bytes_per_col;
                    let mut sumf = 0.0f32;
                    for (b, xb) in x_q8.iter().enumerate().take(blocks_per_col) {
                        let blk_off = col_off + b * 34;
                        let w_scale =
                            half::f16::from_le_bytes([raw[blk_off], raw[blk_off + 1]]).to_f32();
                        let mut sumi: i32 = 0;
                        for i in 0..32 {
                            sumi += (raw[blk_off + 2 + i] as i8) as i32 * xb.qs[i] as i32;
                        }
                        sumf += sumi as f32 * (w_scale * xb.scale);
                    }
                    *yj = sumf;
                }
            }));
        }

        for h in handles {
            h.join().expect("matvec thread panicked");
        }
    });
}

/// SSE4.1-accelerated Q8_0 × Q8_0 integer dot product for a range of output rows.
///
/// Matches GGML's `ggml_vec_dot_q8_0_q8_0`: integer accumulation per block pair,
/// then multiply by both scales. Uses `_mm_cvtepi8_epi16` + `_mm_madd_epi16`
/// for fast i8×i8→i32 accumulation (8 elements per SIMD step).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn matvec_q8_0_sse_range(
    y_chunk: &mut [f32],
    x_q8: &[Q8Block],
    raw: &[u8],
    start_row: usize,
    count: usize,
    blocks_per_col: usize,
    bytes_per_col: usize,
) {
    use std::arch::x86_64::*;

    for (local_j, yj) in y_chunk.iter_mut().enumerate().take(count) {
        let j = start_row + local_j;
        let col_off = j * bytes_per_col;
        let mut sumf = 0.0f32;

        for (b, xb) in x_q8.iter().enumerate().take(blocks_per_col) {
            let blk_off = col_off + b * 34;
            let w_scale =
                half::f16::from_le_bytes([raw[blk_off], raw[blk_off + 1]]).to_f32();
            let w_ptr = raw.as_ptr().add(blk_off + 2);
            let x_qs = &xb.qs;

            // Integer dot product: sum(w_i8[i] * x_i8[i]) for i=0..31
            // Process 8 elements at a time: i8→i16, then madd_epi16 for i16×i16→i32
            let mut acc_i32 = _mm_setzero_si128();

            let mut k = 0;
            while k < 32 {
                // Load 8 weight int8 values, sign-extend to int16
                let w8 = _mm_loadl_epi64(w_ptr.add(k) as *const __m128i);
                let w16 = _mm_cvtepi8_epi16(w8);

                // Load 8 input int8 values, sign-extend to int16
                let x8 = _mm_loadl_epi64(x_qs.as_ptr().add(k) as *const __m128i);
                let x16 = _mm_cvtepi8_epi16(x8);

                // Multiply pairs and add adjacent: (w0*x0 + w1*x1), (w2*x2 + w3*x3), ...
                let prod = _mm_madd_epi16(w16, x16);
                acc_i32 = _mm_add_epi32(acc_i32, prod);

                k += 8;
            }

            // Horizontal sum of 4 × i32 lanes
            let hi = _mm_srli_si128(acc_i32, 8);
            let sum2 = _mm_add_epi32(acc_i32, hi);
            let hi2 = _mm_srli_si128(sum2, 4);
            let sum1 = _mm_add_epi32(sum2, hi2);
            let sumi = _mm_cvtsi128_si32(sum1);

            sumf += sumi as f32 * (w_scale * xb.scale);
        }

        *yj = sumf;
    }
}

/// Scalar Q8_0 × Q8_0 integer dot product fallback
fn matvec_q8_0_scalar(
    y: &mut [f32],
    x_q8: &[Q8Block],
    raw: &[u8],
    n_out: usize,
    blocks_per_col: usize,
    bytes_per_col: usize,
) {
    for (j, yj) in y.iter_mut().enumerate().take(n_out) {
        let col_off = j * bytes_per_col;
        let mut sumf = 0.0f32;
        for (b, xb) in x_q8.iter().enumerate().take(blocks_per_col) {
            let blk_off = col_off + b * 34;
            let w_scale =
                half::f16::from_le_bytes([raw[blk_off], raw[blk_off + 1]]).to_f32();
            let mut sumi: i32 = 0;
            for i in 0..32 {
                sumi += (raw[blk_off + 2 + i] as i8) as i32 * xb.qs[i] as i32;
            }
            sumf += sumi as f32 * (w_scale * xb.scale);
        }
        *yj = sumf;
    }
}

/// SSE4.1-accelerated Q8_0 × Q8_0 integer dot product (single-threaded path).
///
/// Only used for small matvecs (n_out < threshold) that skip threading.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn matvec_q8_0_sse(
    y: &mut [f32],
    x_q8: &[Q8Block],
    raw: &[u8],
    _n_in: usize,
    n_out: usize,
    blocks_per_col: usize,
    bytes_per_col: usize,
) {
    // Delegate to the range function for the full output
    matvec_q8_0_sse_range(y, x_q8, raw, 0, n_out, blocks_per_col, bytes_per_col);
}

/// Compute y[j] = x · W[:,j] for all j, where W is [ne0=n_in, ne1=n_out] stored
/// in GGML column-major F32 format.
pub fn matvec_col_major_f32(y: &mut [f32], x: &[f32], raw: &[u8], n_in: usize, n_out: usize) {
    debug_assert_eq!(x.len(), n_in);
    debug_assert_eq!(y.len(), n_out);
    for (j, yj) in y.iter_mut().enumerate() {
        let col_off = j * n_in;
        let mut sum = 0.0f32;
        for (i, &xi) in x.iter().enumerate() {
            let w_bytes: [u8; 4] = raw[(col_off + i) * 4..(col_off + i) * 4 + 4]
                .try_into()
                .unwrap_or([0u8; 4]);
            sum += f32::from_le_bytes(w_bytes) * xi;
        }
        *yj = sum;
    }
}

/// Dispatch quantized column-major matmul.
/// Handles Q8_0 efficiently; falls back to full dequant for other types.
pub fn matvec_col_major(
    y: &mut [f32],
    x: &[f32],
    raw: &[u8],
    qtype: QuantizationType,
    n_in: usize,
    n_out: usize,
) -> Result<()> {
    match qtype {
        QuantizationType::Q8_0 => {
            if n_in.is_multiple_of(32) {
                matvec_col_major_q8_0(y, x, raw, n_in, n_out);
                return Ok(());
            }
        }
        QuantizationType::F32 => {
            matvec_col_major_f32(y, x, raw, n_in, n_out);
            return Ok(());
        }
        _ => {}
    }
    // Fallback: dequantize all then multiply (may use lots of RAM for large tensors)
    let w_f32 = dequantize(raw, qtype, 1, n_in * n_out)?;
    for j in 0..n_out {
        let mut sum = 0.0f32;
        for i in 0..n_in {
            sum += x[i] * w_f32[j * n_in + i];
        }
        y[j] = sum;
    }
    Ok(())
}

/// Compute selected output columns y[k] = x · W[:, indices[k]] from a GGML
/// column-major weight matrix W without materializing the full output vector.
pub fn matvec_col_major_select(
    y: &mut [f32],
    x: &[f32],
    raw: &[u8],
    qtype: QuantizationType,
    n_in: usize,
    n_out: usize,
    indices: &[usize],
) -> Result<()> {
    debug_assert_eq!(y.len(), indices.len());

    if indices.is_empty() {
        return Ok(());
    }

    match qtype {
        QuantizationType::Q8_0 if n_in.is_multiple_of(32) => {
            let blocks_per_col = n_in / 32;
            let bytes_per_col = blocks_per_col * 34;

            // Quantize input once for all selected columns
            let x_q8 = quantize_f32_to_q8_0(x);

            for (slot, &col_idx) in y.iter_mut().zip(indices.iter()) {
                if col_idx >= n_out {
                    return Err(anyhow!(
                        "selected column {} out of range for output width {}",
                        col_idx,
                        n_out
                    ));
                }

                let col_off = col_idx * bytes_per_col;
                let mut sumf = 0.0f32;
                for (block_idx, xb) in x_q8.iter().enumerate().take(blocks_per_col) {
                    let blk_off = col_off + block_idx * 34;
                    let w_scale =
                        f16::from_le_bytes([raw[blk_off], raw[blk_off + 1]]).to_f32();
                    let mut sumi: i32 = 0;
                    for value_idx in 0..32 {
                        sumi += (raw[blk_off + 2 + value_idx] as i8) as i32 * xb.qs[value_idx] as i32;
                    }
                    sumf += sumi as f32 * (w_scale * xb.scale);
                }
                *slot = sumf;
            }
            Ok(())
        }
        QuantizationType::F32 => {
            for (slot, &col_idx) in y.iter_mut().zip(indices.iter()) {
                if col_idx >= n_out {
                    return Err(anyhow!(
                        "selected column {} out of range for output width {}",
                        col_idx,
                        n_out
                    ));
                }

                let col_off = col_idx * n_in;
                let mut sum = 0.0f32;
                for (row_idx, &xv) in x.iter().enumerate() {
                    let byte_off = (col_off + row_idx) * 4;
                    let weight = f32::from_le_bytes(
                        raw[byte_off..byte_off + 4]
                            .try_into()
                            .map_err(|_| anyhow!("invalid F32 weight bytes"))?,
                    );
                    sum += xv * weight;
                }
                *slot = sum;
            }
            Ok(())
        }
        _ => {
            let w_f32 = dequantize(raw, qtype, 1, n_in * n_out)?;
            for (slot, &col_idx) in y.iter_mut().zip(indices.iter()) {
                if col_idx >= n_out {
                    return Err(anyhow!(
                        "selected column {} out of range for output width {}",
                        col_idx,
                        n_out
                    ));
                }

                let col_off = col_idx * n_in;
                let mut sum = 0.0f32;
                for (row_idx, &xv) in x.iter().enumerate() {
                    sum += xv * w_f32[col_off + row_idx];
                }
                *slot = sum;
            }
            Ok(())
        }
    }
}

/// Compute y = W @ x where x is sparse and only `indices` are non-zero.
///
/// `raw` stores W in GGML column-major format with shape [ne0=n_in, ne1=n_out].
pub fn matvec_col_major_sparse_input(
    y: &mut [f32],
    x: &[f32],
    indices: &[usize],
    raw: &[u8],
    qtype: QuantizationType,
    n_in: usize,
    n_out: usize,
) -> Result<()> {
    debug_assert_eq!(x.len(), indices.len());
    debug_assert_eq!(y.len(), n_out);

    if x.is_empty() {
        y.fill(0.0);
        return Ok(());
    }

    match qtype {
        QuantizationType::Q8_0 if n_in.is_multiple_of(32) => {
            let blocks_per_col = n_in / 32;
            let bytes_per_col = blocks_per_col * 34;

            for (col_idx, slot) in y.iter_mut().enumerate() {
                let col_off = col_idx * bytes_per_col;
                let mut sum = 0.0f32;

                for (&row_idx, &xv) in indices.iter().zip(x.iter()) {
                    if row_idx >= n_in {
                        return Err(anyhow!(
                            "sparse input index {} out of range for input width {}",
                            row_idx,
                            n_in
                        ));
                    }

                    let block_idx = row_idx / 32;
                    let value_idx = row_idx % 32;
                    let blk_off = col_off + block_idx * 34;
                    let scale =
                        f16::from_le_bytes([raw[blk_off], raw[blk_off + 1]]).to_f32();
                    let weight = (raw[blk_off + 2 + value_idx] as i8) as f32 * scale;
                    sum += xv * weight;
                }

                *slot = sum;
            }

            Ok(())
        }
        QuantizationType::F32 => {
            for (col_idx, slot) in y.iter_mut().enumerate() {
                let col_off = col_idx * n_in;
                let mut sum = 0.0f32;

                for (&row_idx, &xv) in indices.iter().zip(x.iter()) {
                    if row_idx >= n_in {
                        return Err(anyhow!(
                            "sparse input index {} out of range for input width {}",
                            row_idx,
                            n_in
                        ));
                    }

                    let byte_off = (col_off + row_idx) * 4;
                    let weight = f32::from_le_bytes(
                        raw[byte_off..byte_off + 4]
                            .try_into()
                            .map_err(|_| anyhow!("invalid F32 weight bytes"))?,
                    );
                    sum += xv * weight;
                }

                *slot = sum;
            }

            Ok(())
        }
        _ => {
            let w_f32 = dequantize(raw, qtype, 1, n_in * n_out)?;
            for (col_idx, slot) in y.iter_mut().enumerate() {
                let col_off = col_idx * n_in;
                let mut sum = 0.0f32;
                for (&row_idx, &xv) in indices.iter().zip(x.iter()) {
                    if row_idx >= n_in {
                        return Err(anyhow!(
                            "sparse input index {} out of range for input width {}",
                            row_idx,
                            n_in
                        ));
                    }
                    sum += xv * w_f32[col_off + row_idx];
                }
                *slot = sum;
            }
            Ok(())
        }
    }
}

/// SwiGLU FFN directly on quantized weight bytes — avoids materializing f32 matrices.
///
/// ffn(x) = down_w @ (SiLU(gate_w @ x) * (up_w @ x))
///
/// All three weight matrices are in GGML column-major format:
/// - gate_raw/up_raw: [ne0=n_embd, ne1=n_ff], maps n_embd → n_ff
/// - down_raw: [ne0=n_ff, ne1=n_embd], maps n_ff → n_embd
#[allow(clippy::too_many_arguments)]
pub fn ffn_swiglu_q(
    out: &mut [f32],
    x: &[f32],
    gate_raw: &[u8],
    up_raw: &[u8],
    down_raw: &[u8],
    qtype: QuantizationType,
    n_embd: usize,
    n_ff: usize,
) -> Result<()> {
    let mut gate = vec![0.0f32; n_ff];
    let mut up = vec![0.0f32; n_ff];

    matvec_col_major(&mut gate, x, gate_raw, qtype, n_embd, n_ff)?;
    matvec_col_major(&mut up, x, up_raw, qtype, n_embd, n_ff)?;

    // SiLU(gate) * up
    for i in 0..n_ff {
        let sig = 1.0 / (1.0 + (-gate[i]).exp());
        gate[i] = gate[i] * sig * up[i];
    }

    // down: n_ff → n_embd
    matvec_col_major(out, &gate, down_raw, qtype, n_ff, n_embd)?;
    Ok(())
}

/// Sparse SwiGLU FFN that only evaluates selected hidden units.
#[allow(clippy::too_many_arguments)]
pub fn ffn_swiglu_q_selected(
    out: &mut [f32],
    x: &[f32],
    gate_raw: &[u8],
    up_raw: &[u8],
    down_raw: &[u8],
    qtype: QuantizationType,
    n_embd: usize,
    n_ff: usize,
    selected_indices: &[usize],
) -> Result<()> {
    if selected_indices.is_empty() {
        out.fill(0.0);
        return Ok(());
    }

    if selected_indices.len() >= n_ff {
        return ffn_swiglu_q(out, x, gate_raw, up_raw, down_raw, qtype, n_embd, n_ff);
    }

    let mut gate = vec![0.0f32; selected_indices.len()];
    let mut up = vec![0.0f32; selected_indices.len()];

    matvec_col_major_select(
        &mut gate,
        x,
        gate_raw,
        qtype,
        n_embd,
        n_ff,
        selected_indices,
    )?;
    matvec_col_major_select(
        &mut up,
        x,
        up_raw,
        qtype,
        n_embd,
        n_ff,
        selected_indices,
    )?;

    for i in 0..selected_indices.len() {
        let sig = 1.0 / (1.0 + (-gate[i]).exp());
        gate[i] = gate[i] * sig * up[i];
    }

    matvec_col_major_sparse_input(out, &gate, selected_indices, down_raw, qtype, n_ff, n_embd)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_f32(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

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
    fn test_matvec_col_major_select_f32() {
        let n_in = 3;
        let n_out = 4;
        let raw = pack_f32(&[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ]);
        let x = [0.5f32, -1.0, 2.0];

        let mut dense = vec![0.0f32; n_out];
        matvec_col_major(&mut dense, &x, &raw, QuantizationType::F32, n_in, n_out).unwrap();

        let indices = [1usize, 3usize];
        let mut selected = vec![0.0f32; indices.len()];
        matvec_col_major_select(
            &mut selected,
            &x,
            &raw,
            QuantizationType::F32,
            n_in,
            n_out,
            &indices,
        )
        .unwrap();

        assert!((selected[0] - dense[1]).abs() < 1e-6);
        assert!((selected[1] - dense[3]).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_col_major_sparse_input_f32() {
        let n_in = 4;
        let n_out = 3;
        let raw = pack_f32(&[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ]);
        let indices = [1usize, 3usize];
        let sparse_x = [2.0f32, -1.0];
        let mut dense_x = vec![0.0f32; n_in];
        dense_x[1] = sparse_x[0];
        dense_x[3] = sparse_x[1];

        let mut expected = vec![0.0f32; n_out];
        matvec_col_major(
            &mut expected,
            &dense_x,
            &raw,
            QuantizationType::F32,
            n_in,
            n_out,
        )
        .unwrap();

        let mut actual = vec![0.0f32; n_out];
        matvec_col_major_sparse_input(
            &mut actual,
            &sparse_x,
            &indices,
            &raw,
            QuantizationType::F32,
            n_in,
            n_out,
        )
        .unwrap();

        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            assert!((*lhs - *rhs).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ffn_swiglu_q_selected_f32() {
        let n_embd = 2;
        let n_ff = 4;
        let gate_raw = pack_f32(&[
            0.5, -0.1,
            1.0, 0.25,
            -0.5, 0.75,
            0.2, 0.4,
        ]);
        let up_raw = pack_f32(&[
            1.0, 0.5,
            -0.25, 0.75,
            0.6, -0.3,
            0.9, 0.1,
        ]);
        let down_raw = pack_f32(&[
            0.1, 0.2, 0.3, 0.4,
            -0.5, 0.25, 0.15, 0.35,
        ]);
        let x = [0.3f32, -0.7];
        let indices = [1usize, 3usize];

        let mut actual = vec![0.0f32; n_embd];
        ffn_swiglu_q_selected(
            &mut actual,
            &x,
            &gate_raw,
            &up_raw,
            &down_raw,
            QuantizationType::F32,
            n_embd,
            n_ff,
            &indices,
        )
        .unwrap();

        let mut gate = vec![0.0f32; n_ff];
        let mut up = vec![0.0f32; n_ff];
        matvec_col_major(&mut gate, &x, &gate_raw, QuantizationType::F32, n_embd, n_ff).unwrap();
        matvec_col_major(&mut up, &x, &up_raw, QuantizationType::F32, n_embd, n_ff).unwrap();

        for idx in 0..n_ff {
            let sig = 1.0 / (1.0 + (-gate[idx]).exp());
            gate[idx] = gate[idx] * sig * up[idx];
            if !indices.contains(&idx) {
                gate[idx] = 0.0;
            }
        }

        let mut expected = vec![0.0f32; n_embd];
        matvec_col_major(
            &mut expected,
            &gate,
            &down_raw,
            QuantizationType::F32,
            n_ff,
            n_embd,
        )
        .unwrap();

        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            assert!((*lhs - *rhs).abs() < 1e-6);
        }
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

    #[test]
    fn test_q4_k_dequantize() {
        let d = f16::from_f32(1.0);
        let dmin = f16::from_f32(0.0);
        let mut block = Vec::new();
        block.extend_from_slice(&d.to_le_bytes()); // d
        block.extend_from_slice(&dmin.to_le_bytes()); // dmin
        block.extend_from_slice(&[0u8; 12]); // scales (12 bytes, all 0)
        block.extend_from_slice(&[0u8; 128]); // qs (128 bytes, all 0)

        let out = dequantize_q4_k(&block, 1, 256).unwrap();
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_q5_k_dequantize() {
        let d = f16::from_f32(1.0);
        let dmin = f16::from_f32(0.0);
        let mut block = Vec::new();
        block.extend_from_slice(&d.to_le_bytes());
        block.extend_from_slice(&dmin.to_le_bytes());
        block.extend_from_slice(&[0u8; 12]); // scales (12 bytes)
        block.extend_from_slice(&[0u8; 32]); // qh (high bits, all 0)
        block.extend_from_slice(&[0u8; 128]); // ql (low bits, all 0)

        let out = dequantize_q5_k(&block, 1, 256).unwrap();
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_q6_k_dequantize() {
        let d = f16::from_f32(1.0);
        let mut block = Vec::new();
        block.extend_from_slice(&d.to_le_bytes()); // d
        block.extend_from_slice(&[0u8; 128]); // ql
        block.extend_from_slice(&[0u8; 64]); // qh
                                             // scales: 18 bytes, all 32 (= 0 after subtracting 32)
        block.extend_from_slice(&[32u8; 18]);

        let out = dequantize_q6_k(&block, 1, 256).unwrap();
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_k_quant_block_sizes() {
        // Verify our block size calculations match the GGUF spec
        assert_eq!(QuantizationType::Q4_K_M.block_size_bytes(), 144);
        assert_eq!(QuantizationType::Q5_K_M.block_size_bytes(), 176);
        assert_eq!(QuantizationType::Q6_K.block_size_bytes(), 212);
        assert_eq!(QuantizationType::Q4_K_M.values_per_block(), 256);
        assert_eq!(QuantizationType::Q5_K_M.values_per_block(), 256);
        assert_eq!(QuantizationType::Q6_K.values_per_block(), 256);
    }

    #[test]
    fn test_get_scale_min_k4() {
        // 8-byte scales array
        // j < 4: d = q[j] & 63, m = q[j+4] & 63
        // j >= 4: d = (q[j+4]&0xF) | ((q[j-4]>>6)<<4)
        //         m = (q[j+4]>>4) | ((q[j]>>6)<<4)
        let scales = [0xFF, 0x00, 0x3F, 0x20, 0x0F, 0x30, 0x01, 0x02];

        // j=0: d = scales[0] & 63 = 0xFF & 63 = 63, m = scales[4] & 63 = 0x0F
        let (d, m) = get_scale_min_k4(0, &scales);
        assert_eq!(d, 63);
        assert_eq!(m, 15);

        // j=4: d = (scales[8] & 0xF) | ((scales[0] >> 6) << 4) — but scales is 8 bytes!
        // So j can only go up to 3 for the first call, then 4-7 for the second
        // Actually: j ranges 0..8 for 8 groups
        // j=4: d = (scales[8] & 0xF) — OUT OF BOUNDS if scales is only 8 bytes
        // But the llama.cpp code accesses q[j+4] for j>=4, meaning q[8..11]
        // This means the scales array in the block is actually larger than 8 bytes

        // The GGUF file stores 12 bytes of scales (matching the C struct)
        // So get_scale_min_k4 needs at least 12 bytes
        let scales12 = [
            0xFF, 0x00, 0x3F, 0x20, 0x0F, 0x30, 0x01, 0x02, 0x0A, 0x0B, 0x0C, 0x0D,
        ];
        let (d, _m) = get_scale_min_k4(4, &scales12);
        // d = (scales[8] & 0xF) | ((scales[0] >> 6) << 4) = (0x0A & 0xF) | ((0xFF >> 6) << 4)
        //   = 0x0A | (3 << 4) = 0x0A | 0x30 = 0x3A = 58
        assert_eq!(d, 58);
    }
}
