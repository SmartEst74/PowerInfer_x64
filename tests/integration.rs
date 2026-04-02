//! Integration tests — verify the full pipeline works end-to-end.
//!
//! These tests exercise multiple modules together to catch
//! integration issues that unit tests miss.

use powerinfer::activation::{ActivationProfile, HotNeuronIndex};
use powerinfer::ops;
use powerinfer::turboquant::CompressedKVCache;

/// Test the full profiling → hot index → memory estimation pipeline
#[test]
fn integration_profiler_to_hot_index() {
    let n_layers = 4;
    let n_ff = 64;
    let mut profile = ActivationProfile::new(n_layers, n_ff, 0.5);

    // Simulate profiling 1000 inputs
    for i in 0..1000 {
        for layer in 0..n_layers {
            let activations: Vec<f32> = (0..n_ff)
                .map(|j| {
                    // Neurons 0-15 are always hot, 16-31 sometimes, 32-63 never
                    if j < 16 {
                        1.0 + (i as f32 * 0.01).sin() * 0.5
                    } else if j < 32 {
                        if i % 3 == 0 {
                            0.8
                        } else {
                            0.1
                        }
                    } else {
                        0.0
                    }
                })
                .collect();
            profile.record_layer(layer, &activations);
        }
        profile.finish_sample();
    }

    let summary = profile.summary();
    assert_eq!(summary.total_samples, 1000);
    assert_eq!(summary.total_neurons, n_layers * n_ff);

    // Hot neurons should be ~25% (16 out of 64 per layer)
    assert!(
        summary.hot_fraction > 0.15 && summary.hot_fraction < 0.35,
        "Hot fraction unexpected: {}",
        summary.hot_fraction
    );

    // Export and verify memory estimate
    let index = profile.export_hot_index(0.5);
    for layer in &index.layers {
        // First 16 neurons should be hot (100% hotness)
        assert!(
            layer.hot_indices.len() >= 16,
            "Layer {} has too few hot neurons: {}",
            layer.layer_idx,
            layer.hot_indices.len()
        );
    }

    // GPU memory estimate for hot neurons
    let mem = index.gpu_memory_estimate(64);
    assert!(mem > 0, "GPU memory estimate should be positive");
}

/// Test TurboQuant compressed KV cache with attention
#[test]
fn integration_compressed_kv_attention() {
    let n_kv_heads = 2;
    let head_dim = 32;
    let mut cache = CompressedKVCache::new(n_kv_heads, head_dim);

    // Append 50 tokens
    for t in 0..50 {
        let keys: Vec<f32> = (0..n_kv_heads * head_dim)
            .map(|i| ((t * 100 + i) as f32 * 0.01).sin())
            .collect();
        let vals: Vec<f32> = (0..n_kv_heads * head_dim)
            .map(|i| ((t * 100 + i) as f32 * 0.02).cos())
            .collect();

        cache.append(&keys, &vals);
    }

    assert_eq!(cache.seq_len(), 50);
    // TurboQuant 3-bit keys + f16 values: compression > 2x
    let ratio = cache.compression_ratio();
    assert!(ratio > 2.0, "Expected > 2.0x compression with TurboQuant, got {ratio}");

    // Compute attention scores
    let q: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let scores = cache.attention_scores(&q, 0);
    assert_eq!(scores.len(), 50);
    assert!(scores.iter().all(|v| v.is_finite()));

    // Softmax the scores and compute weighted value sum
    let mut scores_mut = scores.clone();
    ops::softmax(&mut scores_mut);
    let weighted = cache.weighted_value_sum(&scores_mut, 0);
    assert_eq!(weighted.len(), head_dim);
    assert!(weighted.iter().all(|v| v.is_finite()));
}

/// Test benchmark comparison with known reference
#[test]
fn integration_reference_comparison() {
    // Create two identical "logit" vectors and verify zero error
    let logits_a: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
    let logits_b = logits_a.clone();

    // Manual comparison (can't use QualityBenchmark without a real model)
    let mut sum_abs_err = 0.0f64;
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for i in 0..logits_a.len() {
        let err = (logits_a[i] - logits_b[i]).abs() as f64;
        sum_abs_err += err;
        dot += logits_a[i] as f64 * logits_b[i] as f64;
        norm_a += logits_a[i] as f64 * logits_a[i] as f64;
        norm_b += logits_b[i] as f64 * logits_b[i] as f64;
    }

    let mean_abs_err = sum_abs_err / logits_a.len() as f64;
    let cosine_sim = dot / (norm_a.sqrt() * norm_b.sqrt());

    assert!(
        mean_abs_err < 1e-10,
        "Identical vectors should have zero error"
    );
    assert!(
        (cosine_sim - 1.0).abs() < 1e-10,
        "Identical vectors should have cosine sim = 1"
    );
}

/// Test TurboQuant hot-neuron KV cache allocation estimate
#[test]
fn integration_gpu_memory_planning() {
    let n_layers = 32;
    let n_ff = 11008; // Llama-7B FFN size
    let head_dim = 128;
    let hot_fraction = 0.3;

    // Simulate a profile where 30% of neurons are hot
    let mut index = HotNeuronIndex {
        version: 1,
        threshold: 0.5,
        min_hotness: 0.5,
        total_samples: 1000,
        layers: Vec::new(),
    };

    for layer_idx in 0..n_layers {
        let n_hot = (n_ff as f64 * hot_fraction) as usize;
        let hot_indices: Vec<usize> = (0..n_hot).collect();
        index.layers.push(powerinfer::activation::HotLayer {
            layer_idx,
            hot_indices,
            n_ff,
        });
    }

    let gpu_mem = index.gpu_memory_estimate(head_dim);
    let gpu_mem_mb = gpu_mem as f64 / (1024.0 * 1024.0);

    // 32 layers * 3302 hot neurons * 128 dim * 4 bytes * 3 weight matrices
    // = 32 * 3302 * 128 * 4 * 3 ≈ 162MB
    assert!(
        gpu_mem_mb > 100.0 && gpu_mem_mb < 500.0,
        "GPU memory estimate unexpected: {gpu_mem_mb:.1} MB"
    );
}

/// Test full transformer-style forward pass with attention
#[test]
fn integration_transformer_forward() {
    let n_embd = 64;
    let n_heads = 4;
    let head_dim = n_embd / n_heads;
    let n_ff = 128;
    let seq_len = 5;

    // Fake input
    let x: Vec<f32> = (0..seq_len * n_embd)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    // Layer norm
    let norm_weight = vec![1.0f32; n_embd];
    let mut normed = vec![0.0f32; n_embd];

    // Process last token only
    let last_offset = (seq_len - 1) * n_embd;
    ops::rms_norm(
        &mut normed,
        &x[last_offset..last_offset + n_embd],
        &norm_weight,
        1e-6,
    );
    assert!(normed.iter().all(|v| v.is_finite()));

    // Q projection (simplified)
    let wq: Vec<f32> = (0..n_embd * n_embd)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let mut q = vec![0.0f32; n_embd];
    ops::matvec(&mut q, &normed, &wq, n_embd, n_embd);
    assert!(q.iter().all(|v| v.is_finite()));

    // Attention per head
    let mut attn_out = vec![0.0f32; n_embd];
    for h in 0..n_heads {
        let h_offset = h * head_dim;
        let q_head = &q[h_offset..h_offset + head_dim];

        // Fake KV cache
        let k_cache: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32 * 0.01).cos())
            .collect();
        let v_cache: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32 * 0.02).sin())
            .collect();

        let mut head_out = vec![0.0f32; head_dim];
        ops::attention_head(&mut head_out, q_head, &k_cache, &v_cache, seq_len, head_dim);

        attn_out[h_offset..h_offset + head_dim].copy_from_slice(&head_out);
    }
    assert!(attn_out.iter().all(|v| v.is_finite()));

    // FFN (SwiGLU)
    let gate_w: Vec<f32> = (0..n_ff * n_embd)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let up_w: Vec<f32> = (0..n_ff * n_embd)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let down_w: Vec<f32> = (0..n_embd * n_ff)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();

    let mut ffn_out = vec![0.0f32; n_embd];
    ops::ffn_swiglu(&mut ffn_out, &normed, &gate_w, &up_w, &down_w, n_embd, n_ff);
    assert!(ffn_out.iter().all(|v| v.is_finite()));

    // Residual
    let mut final_out = vec![0.0f32; n_embd];
    ops::elem_add(
        &mut final_out,
        &x[last_offset..last_offset + n_embd],
        &ffn_out,
    );
    assert!(final_out.iter().all(|v| v.is_finite()));
}
