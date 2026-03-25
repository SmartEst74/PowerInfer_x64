//! Half-precision (f16) utilities for quant operations

use half::f16;

/// Convert f32 to f16 with saturation
pub fn f32_to_f16(x: f32) -> f16 {
    f16::from_f32(x)
}

/// Convert f16 to f32
pub fn f16_to_f32(x: f16) -> f32 {
    x.to_f32()
}

/// Multiply two f16 values, return f32 for accumulation
#[inline(always)]
pub fn mul_f16(a: f16, b: f16) -> f32 {
    a.to_f32() * b.to_f32()
}

/// Add f16 to f32 accumulator (converts f16→f32)
#[inline(always)]
pub fn add_f16_to_f32(acc: f32, a: f16) -> f32 {
    acc + a.to_f32()
}

/// Convert f32 sum to f16 with rounding
#[inline(always)]
pub fn f32_to_f16_round(sum: f32) -> f16 {
    f16::from_f32(sum)
}
