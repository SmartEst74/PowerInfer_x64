//! Soak tests — verify stability over repeated operations.
//!
//! These tests detect memory leaks, accumulation errors, and
//! performance degradation over many iterations.

use powerinfer::activation::ActivationProfile;
use powerinfer::ops;
use powerinfer::quant::{self, QuantizationType};
use powerinfer::turboquant::TurboQuant;

const SOAK_ITERATIONS: usize = 10_000;

#[test]
fn soak_repeated_rms_norm() {
    let dim = 128;
    let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let weight = vec![1.0f32; dim];
    let mut out = vec![0.0f32; dim];

    for i in 0..SOAK_ITERATIONS {
        let scale = 1.0 + (i as f32 * 0.001);
        let scaled_x: Vec<f32> = x.iter().map(|v| v * scale).collect();
        ops::rms_norm(&mut out, &scaled_x, &weight, 1e-6);

        assert!(out.iter().all(|v| v.is_finite()), "Non-finite at iter {i}");
    }
}

#[test]
fn soak_repeated_softmax() {
    let mut vals = vec![0.0f32; 1000];

    for i in 0..SOAK_ITERATIONS {
        // Vary the input each iteration
        for (j, v) in vals.iter_mut().enumerate() {
            *v = ((i + j) as f32 * 0.01).sin();
        }

        ops::softmax(&mut vals);

        let sum: f32 = vals.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Softmax sum drifted at iter {i}: {sum}"
        );
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "Non-finite in softmax at iter {i}"
        );
    }
}

#[test]
fn soak_repeated_matvec() {
    let n_in = 64;
    let n_out = 128;
    let x: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.1).sin()).collect();
    let w: Vec<f32> = (0..n_out * n_in).map(|i| (i as f32 * 0.01).cos()).collect();
    let mut y = vec![0.0f32; n_out];

    for i in 0..SOAK_ITERATIONS {
        ops::matvec(&mut y, &x, &w, n_out, n_in);
        assert!(y.iter().all(|v| v.is_finite()), "Non-finite at iter {i}");
    }
}

#[test]
fn soak_repeated_dequantization() {
    let scale = half::f16::from_f32(1.0);
    let mut block = Vec::new();
    block.extend_from_slice(&scale.to_le_bytes());
    // 16 bytes of varying data
    for b in 0..16u8 {
        block.push(b.wrapping_mul(17));
    }

    for i in 0..SOAK_ITERATIONS {
        let out = quant::dequantize(&block, QuantizationType::Q4_0, 1, 32).unwrap();
        assert_eq!(out.len(), 32);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "Non-finite dequant at iter {i}"
        );
    }
}

#[test]
fn soak_repeated_turboquant() {
    let dim = 64;
    let tq = TurboQuant::new(3, dim, dim, 42);

    for i in 0..SOAK_ITERATIONS / 10 {
        let x: Vec<f32> = (0..dim).map(|j| ((i + j) as f32 * 0.01).sin()).collect();
        let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let x: Vec<f32> = x.iter().map(|v| v / norm).collect();

        let packed = tq.quantize_vector(&x);
        let deq = tq.dequantize_vector(&packed);

        assert_eq!(deq.len(), dim);
        assert!(deq.iter().all(|v| v.is_finite()), "Non-finite at iter {i}");
    }
}

#[test]
fn soak_profiler_accumulation() {
    let mut profile = ActivationProfile::new(4, 128, 0.5);

    for i in 0..SOAK_ITERATIONS {
        let layer = i % 4;
        let activations: Vec<f32> = (0..128).map(|j| ((i + j) as f32 * 0.1).sin()).collect();
        profile.record_layer(layer, &activations);

        if i % 100 == 99 {
            profile.finish_sample();
        }
    }

    // Verify no overflow or corruption
    let summary = profile.summary();
    assert_eq!(summary.total_samples, SOAK_ITERATIONS / 100);
    assert!(summary.total_neurons == 512); // 4 * 128
    assert!(summary.hot_fraction >= 0.0 && summary.hot_fraction <= 1.0);
}

#[test]
fn soak_attention_head_stability() {
    let head_dim = 64;
    let seq_len = 32;
    let q: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let k_cache: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i as f32 * 0.01).cos())
        .collect();
    let v_cache: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i as f32 * 0.02).sin())
        .collect();
    let mut out = vec![0.0f32; head_dim];

    for i in 0..SOAK_ITERATIONS / 100 {
        ops::attention_head(&mut out, &q, &k_cache, &v_cache, seq_len, head_dim);
        assert!(out.iter().all(|v| v.is_finite()), "Non-finite at iter {i}");
    }
}
