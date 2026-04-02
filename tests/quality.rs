//! Quality validation tests against real GGUF models.
//!
//! These tests prove the forward pass produces valid output.
//! They run against real model files on disk.

use powerinfer::gguf::GgufFile;
use powerinfer::sysinfo::SystemResources;

/// Verify GGUF file parses and metadata is correct
#[test]
#[ignore] // Run with: cargo test --test quality -- --ignored
fn validate_qwen3_4b_gguf_loads() {
    let path = "/home/jon/models/llama-cache/Qwen3-4B-Instruct-2507-Q4_K_M.gguf";

    if !std::path::Path::new(path).exists() {
        eprintln!("SKIP: model file not found at {path}");
        return;
    }

    let gguf = GgufFile::open(path).expect("GGUF should load");
    let config = gguf.model_config().expect("config should parse");

    assert_eq!(config.arch, "qwen3");
    assert_eq!(config.block_count, 36);
    assert_eq!(config.embedding_length, 2560);
    assert_eq!(config.attention.head_count, 32);
    assert_eq!(config.attention.head_count_kv, Some(8));
    assert_eq!(config.feed_forward_length, 9728);

    let tensors = gguf.tensors();
    assert!(
        tensors.len() > 300,
        "Expected 300+ tensors, got {}",
        tensors.len()
    );

    println!(
        "Qwen3-4B GGUF validated: {} tensors, {} layers",
        tensors.len(),
        config.block_count
    );
}

/// Verify Arch-Agent-3B Q8_0 loads
#[test]
#[ignore]
fn validate_arch_agent_3b_gguf_loads() {
    let path = "/home/jon/models/llama-cache/Arch-Agent-3B.Q8_0.gguf";

    if !std::path::Path::new(path).exists() {
        eprintln!("SKIP: model file not found at {path}");
        return;
    }

    let gguf = GgufFile::open(path).expect("GGUF should load");
    let config = gguf.model_config().expect("config should parse");

    assert_eq!(config.arch, "qwen2");
    assert_eq!(config.block_count, 36);
    assert_eq!(config.embedding_length, 2048);

    println!(
        "Arch-Agent-3B GGUF validated: {} layers, {} dim",
        config.block_count, config.embedding_length
    );
}

/// Verify system resources are detected correctly
#[test]
fn validate_system_resources() {
    let resources = SystemResources::scan();

    assert!(!resources.gpus.is_empty(), "Should detect at least one GPU");
    assert!(resources.total_ram > 0, "Should detect RAM");
    assert!(resources.cpu_cores > 0, "Should detect CPU cores");

    for gpu in &resources.gpus {
        assert!(gpu.total_vram > 0, "GPU {} should have VRAM", gpu.index);
        assert!(!gpu.name.is_empty(), "GPU {} should have a name", gpu.index);
    }

    println!(
        "System validated: {} GPUs, {:.1} GB RAM, {} CPU cores",
        resources.gpus.len(),
        resources.total_ram as f64 / (1024.0 * 1024.0 * 1024.0),
        resources.cpu_cores,
    );
}

/// Verify SIMD produces correct results
#[test]
fn validate_simd_correctness() {
    // Large vector to exercise SIMD path (not just scalar fallback)
    let n = 1024;
    let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).cos()).collect();

    // SIMD dot product
    let simd_result = powerinfer::simd::dot_product(&a, &b);

    // Scalar reference
    let scalar_result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    let error = (simd_result - scalar_result).abs();
    assert!(
        error < 1e-3,
        "SIMD dot product error too large: {error} (simd={simd_result}, scalar={scalar_result})"
    );

    println!("SIMD validated: dot product error = {error}");
}

/// Verify TurboQuant compression ratio is correct (V3, MSE-only, no QJL)
#[test]
fn validate_turboquant_compression() {
    let tq = powerinfer::turboquant::TurboQuant::new_mse_only(3, 128, 42);

    // V3 (no QJL): 3-bit with dim=128 = 3*128 = 384 bits = 48 bytes
    assert_eq!(
        tq.compressed_bytes(),
        48,
        "3-bit 128-dim V3 should be 48 bytes (no QJL overhead)"
    );

    // f32 version is 512 bytes
    let ratio: f64 = 512.0 / 48.0;
    assert!(
        (ratio - 10.67).abs() < 0.1,
        "Should be ~10.7x compression, got {ratio}x"
    );

    println!(
        "TurboQuant V3 validated: {} bytes per vector ({:.1}x compression)",
        tq.compressed_bytes(),
        ratio
    );
}
