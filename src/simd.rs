//! SIMD-accelerated operations using SSE (128-bit, 4 floats at a time)
//!
//! Provides 4-8x speedup over scalar operations on x86_64 CPUs with SSE.
//! Uses std::arch::x86_64 SSE intrinsics.
//!
//! Target: Pentium G4400 (SSE4.1/SSE4.2, no AVX)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD dot product: sum of element-wise multiplication of a and b
/// Uses SSE to process 4 floats at a time.
/// Safety: requires SSE4.1 CPU support. Caller must verify via is_x86_feature_detected!
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn dot_product_sse(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    // Process 4 floats at a time
    while i + 4 <= n {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
        i += 4;
    }

    // Horizontal sum of the 4 elements in sum
    let mut result = hsum_sse(sum);

    // Handle remaining elements
    while i < n {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

/// Horizontal sum of a 128-bit SSE register containing 4 f32 values
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn hsum_sse(v: __m128) -> f32 {
    // Shuffle and add: [a, b, c, d] -> [a+b, c+d, a+b, c+d]
    let shuf = _mm_movehdup_ps(v);
    let sums = _mm_add_ps(v, shuf);
    // [a+b+c+d, _, _, _]
    let shuf2 = _mm_movehl_ps(sums, sums);
    let final_sum = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(final_sum)
}

/// SIMD matrix-vector multiplication: y = x @ w
/// where x is [n_in], w is [n_out, n_in] (row-major), y is [n_out]
/// Each row of w is dot-producted with x using SSE.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn matvec_sse(y: &mut [f32], x: &[f32], w: &[f32], n_out: usize, n_in: usize) {
    let w_len = w.len();
    for (i, yi) in y.iter_mut().enumerate().take(n_out) {
        let row_offset = i * n_in;
        if row_offset + n_in > w_len {
            break;
        }
        let row = &w[row_offset..row_offset + n_in];
        *yi = dot_product_sse(x, row);
    }
}

/// SIMD element-wise addition: out = a + b
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn elem_add_sse(out: &mut [f32], a: &[f32], b: &[f32]) {
    let n = out.len();
    let mut i = 0;

    while i + 4 <= n {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let result = _mm_add_ps(va, vb);
        _mm_storeu_ps(out.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < n {
        out[i] = a[i] + b[i];
        i += 1;
    }
}

/// SIMD element-wise multiplication: out = a * b
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn elem_mul_sse(out: &mut [f32], a: &[f32], b: &[f32]) {
    let n = out.len();
    let mut i = 0;

    while i + 4 <= n {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let result = _mm_mul_ps(va, vb);
        _mm_storeu_ps(out.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < n {
        out[i] = a[i] * b[i];
        i += 1;
    }
}

/// SIMD scalar multiply and add: out = a * scalar + b (FMA-like without AVX)
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn scale_add_sse(out: &mut [f32], a: &[f32], scalar: f32, b: &[f32]) {
    let n = out.len();
    let vs = _mm_set1_ps(scalar);
    let mut i = 0;

    while i + 4 <= n {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let result = _mm_add_ps(_mm_mul_ps(va, vs), vb);
        _mm_storeu_ps(out.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < n {
        out[i] = a[i] * scalar + b[i];
        i += 1;
    }
}

/// SIMD sum of squares: sum(a[i] * a[i])
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn sum_squares_sse(a: &[f32]) -> f32 {
    let n = a.len();
    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= n {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        sum = _mm_add_ps(sum, _mm_mul_ps(va, va));
        i += 4;
    }

    let mut result = hsum_sse(sum);
    while i < n {
        result += a[i] * a[i];
        i += 1;
    }

    result
}

/// SIMD scale: out = a * scalar
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn scale_sse(out: &mut [f32], a: &[f32], scalar: f32) {
    let n = out.len();
    let vs = _mm_set1_ps(scalar);
    let mut i = 0;

    while i + 4 <= n {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let result = _mm_mul_ps(va, vs);
        _mm_storeu_ps(out.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < n {
        out[i] = a[i] * scalar;
        i += 1;
    }
}

/// Safe wrapper for SIMD matvec (falls back to scalar if not x86_64)
pub fn matvec(y: &mut [f32], x: &[f32], w: &[f32], n_out: usize, n_in: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            unsafe {
                matvec_sse(y, x, w, n_out, n_in);
            }
            return;
        }
    }
    // Fallback to scalar
    crate::quant::matvec_f32(y, x, w, n_out, n_in);
}

/// Safe wrapper for SIMD dot product
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            return unsafe { dot_product_sse(a, b) };
        }
    }
    // Fallback
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Safe wrapper for SIMD element-wise add
pub fn elem_add(out: &mut [f32], a: &[f32], b: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            unsafe {
                elem_add_sse(out, a, b);
            }
            return;
        }
    }
    for ((o, &ai), &bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = ai + bi;
    }
}

/// Safe wrapper for SIMD element-wise multiply
pub fn elem_mul(out: &mut [f32], a: &[f32], b: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            unsafe {
                elem_mul_sse(out, a, b);
            }
            return;
        }
    }
    for ((o, &ai), &bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = ai * bi;
    }
}

/// Safe wrapper for SIMD sum of squares
pub fn sum_squares(a: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            return unsafe { sum_squares_sse(a) };
        }
    }
    a.iter().map(|v| v * v).sum()
}

/// Safe wrapper for SIMD scale
pub fn scale(out: &mut [f32], a: &[f32], scalar: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            unsafe {
                scale_sse(out, a, scalar);
            }
            return;
        }
    }
    for (o, &ai) in out.iter_mut().zip(a.iter()) {
        *o = ai * scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0f32];
        // 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1 = 120
        let result = dot_product(&a, &b);
        assert!(
            (result - 120.0).abs() < 1e-5,
            "Expected 120.0, got {result}"
        );
    }

    #[test]
    fn test_dot_product_unaligned() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0f32]; // not a multiple of 4
        let b = [5.0, 4.0, 3.0, 2.0, 1.0f32];
        // 1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 35
        let result = dot_product(&a, &b);
        assert!((result - 35.0).abs() < 1e-5, "Expected 35.0, got {result}");
    }

    #[test]
    fn test_matvec() {
        // [1, 2] @ [[3, 4], [5, 6]] = [11, 17]
        let x = [1.0f32, 2.0];
        let w = [3.0f32, 4.0, 5.0, 6.0];
        let mut y = [0.0f32; 2];
        matvec(&mut y, &x, &w, 2, 2);
        assert!((y[0] - 11.0).abs() < 1e-5, "Expected 11.0, got {}", y[0]);
        assert!((y[1] - 17.0).abs() < 1e-5, "Expected 17.0, got {}", y[1]);
    }

    #[test]
    fn test_elem_add() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0f32];
        let b = [5.0, 4.0, 3.0, 2.0, 1.0f32];
        let mut out = [0.0f32; 5];
        elem_add(&mut out, &a, &b);
        assert_eq!(out, [6.0, 6.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_elem_mul() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0f32];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0f32];
        let mut out = [0.0f32; 5];
        elem_mul(&mut out, &a, &b);
        assert_eq!(out, [2.0, 6.0, 12.0, 20.0, 30.0]);
    }

    #[test]
    fn test_sum_squares() {
        let a = [1.0, 2.0, 3.0, 4.0f32];
        // 1 + 4 + 9 + 16 = 30
        let result = sum_squares(&a);
        assert!((result - 30.0).abs() < 1e-5, "Expected 30.0, got {result}");
    }

    #[test]
    fn test_scale() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0f32];
        let mut out = [0.0f32; 5];
        scale(&mut out, &a, 2.0);
        assert_eq!(out, [2.0, 4.0, 6.0, 8.0, 10.0]);
    }
}
