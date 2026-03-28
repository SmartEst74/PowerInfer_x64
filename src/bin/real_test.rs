//! Real model integration test binary.
//!
//! Loads a GGUF model, runs forward pass, and reports results.
//! Usage: cargo run --bin real_test -- <path_to.gguf>

use std::env;
use std::time::Instant;

use powerinfer::gguf::GgufFile;
use powerinfer::model::InferenceContext;
use powerinfer::runtime::BackendFactory;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let model_path = env::args().nth(1).unwrap_or_else(|| {
        "/home/jon/models/llama-cache/Qwen3-4B-Instruct-2507-Q4_K_M.gguf".to_string()
    });

    println!("=== PowerInfer_x64 Real Model Test ===");
    println!("Model: {model_path}");
    println!();

    // Step 1: Load GGUF
    println!("[1/5] Loading GGUF file...");
    let start = Instant::now();
    let gguf = match GgufFile::open(&model_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("FAILED to load GGUF: {e}");
            std::process::exit(1);
        }
    };
    println!("  Loaded in {:.2}s", start.elapsed().as_secs_f64());

    // Step 2: Read metadata
    println!("[2/5] Reading model metadata...");
    let config = match gguf.model_config() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FAILED to read config: {e}");
            std::process::exit(1);
        }
    };
    println!("  Architecture: {}", config.arch);
    println!("  Name: {}", config.name.as_deref().unwrap_or("unknown"));
    println!("  Layers: {}", config.block_count);
    println!("  Embedding dim: {}", config.embedding_length);
    println!("  FFN dim: {}", config.feed_forward_length);
    println!("  Attention heads: {}", config.attention.head_count);
    println!(
        "  KV heads: {}",
        config.attention.head_count_kv.unwrap_or(0)
    );
    println!("  Head dim: {}", config.attention.head_dim);
    println!("  Context length: {}", config.context_length);
    println!("  Quantization: {:?}", config.quantization);
    println!(
        "  Parameters: {}",
        gguf.parameter_count().unwrap_or("unknown")
    );
    if let Some(moe) = config.moe {
        println!("  MoE experts: {}", moe.expert_count);
        println!("  MoE active experts: {}", moe.expert_used_count);
    }
    println!();

    // Step 3: List tensors
    println!("[3/5] Listing tensors...");
    let tensors = gguf.tensors();
    println!("  Total tensors: {}", tensors.len());
    println!("  First 10:");
    for t in tensors.iter().take(10) {
        let shape: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
        println!("    {} [{:?}] kind={}", t.name, shape.join(", "), t.kind);
    }
    if tensors.len() > 10 {
        println!("    ... and {} more", tensors.len() - 10);
    }
    println!();

    // Step 4: Build inference context (loads weights)
    println!("[4/5] Building inference context (loading weights)...");
    let start = Instant::now();
    let mut ctx = match InferenceContext::from_gguf(&model_path, BackendFactory::cpu()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FAILED to build context: {e}");
            eprintln!("This is expected — we need to fix weight loading for this model.");
            eprintln!(
                "The GGUF was parsed successfully. Progress: model loads, weights need work."
            );
            std::process::exit(1);
        }
    };
    println!("  Built in {:.2}s", start.elapsed().as_secs_f64());
    println!("  Backend: {}", ctx.backend_name());

    // Step 5: Run inference
    println!("[5/5] Running inference...");
    let prompt = "The capital of France is";
    println!("  Prompt: \"{prompt}\"");

    let start = Instant::now();
    match ctx.generate(prompt, 10) {
        Ok(output) => {
            let elapsed = start.elapsed().as_secs_f64();
            println!("  Output: \"{output}\"");
            println!("  Time: {:.2}s", elapsed);
            println!();
            println!("=== SUCCESS ===");
            println!("Model loaded, inference ran, output produced.");
        }
        Err(e) => {
            eprintln!("  FAILED: {e}");
            eprintln!();
            eprintln!("The model loaded but inference failed.");
            eprintln!("This needs debugging — but the GGUF parsing works.");
            std::process::exit(1);
        }
    }

    Ok(())
}
