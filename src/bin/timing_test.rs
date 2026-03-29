//! Debug test: run a single forward pass with timing to find the bottleneck.
//! Usage: cargo run --bin timing_test -- <model.gguf>

use std::env;
use std::time::Instant;

use powerinfer::gguf::GgufFile;
use powerinfer::model::InferenceContext;
use powerinfer::ops;
use powerinfer::runtime::BackendFactory;
use powerinfer::sysinfo::SystemResources;
use powerinfer::weights::Weights;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let model_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jon/models/llama-cache/Arch-Agent-3B.Q8_0.gguf".to_string());

    println!("=== Timing Debug Test ===\n");

    // Show system resources
    let resources = SystemResources::scan();
    resources.print_summary();
    println!();

    // Load GGUF
    println!("[1] Loading GGUF...");
    let start = Instant::now();
    let gguf = GgufFile::open(&model_path)?;
    let config = gguf.model_config()?;
    println!("  Loaded in {:.2}s", start.elapsed().as_secs_f64());
    println!(
        "  Arch: {}, Layers: {}, Dim: {}, FFN: {}",
        config.arch, config.block_count, config.embedding_length, config.feed_forward_length
    );

    // Load weights
    println!("[2] Loading weights...");
    let start = Instant::now();
    let weights = Weights::from_gguf(&gguf)?;
    println!(
        "  Loaded in {:.2}s ({} tensors)",
        start.elapsed().as_secs_f64(),
        weights.len()
    );

    // Test single matvec performance
    println!("[3] Benchmarking SIMD matvec...");
    let n_embd = config.embedding_length;
    let n_ff = config.feed_forward_length;
    let x: Vec<f32> = (0..n_embd).map(|i| (i as f32 * 0.001).sin()).collect();
    let w: Vec<f32> = (0..n_embd * n_ff)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let mut y = vec![0.0f32; n_ff];

    let start = Instant::now();
    let iterations = 100;
    for _ in 0..iterations {
        ops::matvec(&mut y, &x, &w, n_ff, n_embd);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let per_matvec_ms = elapsed / iterations as f64 * 1000.0;
    println!("  matvec({n_embd}→{n_ff}): {per_matvec_ms:.2}ms per call ({iterations} iterations)");

    // Estimate forward pass time
    let matvecs_per_layer = 7; // q, k, v, attn_out, ffn_gate, ffn_up, ffn_down
    let layer_time_ms = per_matvec_ms * matvecs_per_layer as f64;
    let total_time_s = layer_time_ms * config.block_count as f64 / 1000.0;
    println!("  Estimated forward pass: {total_time_s:.1}s ({layer_time_ms:.1}ms per layer)");

    // Try a single forward pass with just 1 token
    println!("[4] Single forward pass (1 token)...");
    let tokens: Vec<u32> = vec![0]; // just token 0

    let start = Instant::now();
    let mut ctx = InferenceContext::from_gguf(&model_path, BackendFactory::cpu())?;
    let build_time = start.elapsed().as_secs_f64();
    println!("  Context built in {build_time:.2}s");

    let start = Instant::now();
    match ctx.forward(&tokens) {
        Ok(logits) => {
            let elapsed = start.elapsed().as_secs_f64();
            println!("  Forward pass: {elapsed:.2}s");
            println!("  Output logits: {} values", logits.len());
            // Show top-5 logits
            let mut indexed: Vec<(f32, usize)> =
                logits.iter().enumerate().map(|(i, &v)| (v, i)).collect();
            indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            println!("  Top-5 logits:");
            for (i, (val, idx)) in indexed.iter().take(5).enumerate() {
                println!("    {}: token {idx} = {val:.4}", i + 1);
            }
            println!(
                "  All logits finite: {}",
                logits.iter().all(|v| v.is_finite())
            );
            println!("\n=== SUCCESS: forward pass produced output ===");
        }
        Err(e) => {
            let elapsed = start.elapsed().as_secs_f64();
            println!("  FAILED after {elapsed:.2}s: {e}");
        }
    }

    Ok(())
}
