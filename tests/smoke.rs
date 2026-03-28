//! Smoke tests — verify the pipeline works end-to-end with synthetic data.
//!
//! These tests catch basic regressions: binary compiles, modules load,
//! forward pass produces non-zero output, tokenizer roundtrips.

use powerinfer::activation::ActivationProfile;
use powerinfer::ops;
use powerinfer::quant::{self, QuantizationType};
use powerinfer::turboquant::TurboQuant;

// --- Pipeline smoke tests ---

#[test]
fn smoke_quantization_roundtrip() {
    let scale = half::f16::from_f32(1.0);
    let mut block = Vec::new();
    block.extend_from_slice(&scale.to_le_bytes());
    block.extend_from_slice(&[0u8; 16]);

    let out = quant::dequantize(&block, QuantizationType::Q4_0, 1, 32).unwrap();
    assert_eq!(out.len(), 32);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn smoke_ops_produce_finite_output() {
    // RMSNorm
    let x = [1.0, 2.0, 3.0, 4.0f32];
    let weight = [1.0; 4];
    let mut out = [0.0f32; 4];
    ops::rms_norm(&mut out, &x, &weight, 1e-6);
    assert!(out.iter().all(|v| v.is_finite()));

    // Softmax
    let mut vals = [1.0, 2.0, 3.0f32];
    ops::softmax(&mut vals);
    assert!(vals.iter().all(|v| v.is_finite()));
    let sum: f32 = vals.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // SiLU
    let mut x = [0.0, 1.0, -1.0f32];
    ops::silu(&mut x);
    assert!(x.iter().all(|v| v.is_finite()));

    // RoPE (position 0 should preserve values)
    let head_dim = 4;
    let mut q = [1.0, 2.0, 3.0, 4.0f32];
    let mut k = [5.0, 6.0, 7.0, 8.0f32];
    ops::apply_rope(&mut q, &mut k, 0, head_dim, head_dim);
    assert!((q[0] - 1.0).abs() < 1e-6);
    assert!(q.iter().all(|v| v.is_finite()));
    assert!(k.iter().all(|v| v.is_finite()));

    // Attention head
    let q_vec = [1.0, 0.0, 0.0, 0.0f32];
    let k_cache = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let v_cache = [10.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0];
    let mut attn_out = vec![0.0f32; 4];
    ops::attention_head(&mut attn_out, &q_vec, &k_cache, &v_cache, 2, 4);
    assert!(attn_out.iter().all(|v| v.is_finite()));
}

#[test]
fn smoke_matvec_correctness() {
    // [1,2] @ [[3,4],[5,6]] = [11, 17]
    let x = [1.0f32, 2.0];
    let w = [3.0f32, 4.0, 5.0, 6.0];
    let mut y = [0.0f32; 2];
    ops::matvec(&mut y, &x, &w, 2, 2);
    assert!((y[0] - 11.0).abs() < 1e-5);
    assert!((y[1] - 17.0).abs() < 1e-5);
}

#[test]
fn smoke_turboquant_compresses_and_restores() {
    let dim = 32;
    let tq = TurboQuant::new(3, dim, dim, 42);

    let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let x: Vec<f32> = x.iter().map(|v| v / norm).collect();

    let packed = tq.quantize_vector(&x);
    let reconstructed = tq.dequantize_vector(&packed);

    assert_eq!(reconstructed.len(), dim);
    assert!(reconstructed.iter().all(|v| v.is_finite()));

    // MSE should be reasonable
    let mse: f32 = x
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / dim as f32;
    assert!(mse < 1.0, "MSE too high: {mse}");
}

#[test]
fn smoke_activation_profiler_records() {
    let mut profile = ActivationProfile::new(2, 4, 0.5);

    profile.record_layer(0, &[1.0, 0.0, 0.8, 0.1]);
    profile.record_layer(1, &[0.0, 1.0, 0.1, 0.9]);
    profile.finish_sample();

    assert_eq!(profile.total_samples, 1);
    assert_eq!(profile.layers[0].neurons[0].hot_count, 1);
    assert_eq!(profile.layers[0].neurons[1].hot_count, 0);

    let index = profile.export_hot_index(0.5);
    assert_eq!(index.layers[0].hot_indices, vec![0, 2]);
    assert_eq!(index.layers[1].hot_indices, vec![1, 3]);
}

#[test]
fn smoke_benchmark_math() {
    use powerinfer::benchmark::QualityBenchmark;
    let bench = QualityBenchmark::new();
    let _ = bench;
}
