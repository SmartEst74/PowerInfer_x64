//! Debug test: inspect the actual logits from a forward pass.
//! Usage: cargo run --release --bin debug_logits

use powerinfer::model::InferenceContext;
use powerinfer::runtime::BackendFactory;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let model_path = "/home/jon/models/llama-cache/Arch-Agent-3B.Q8_0.gguf";

    let start = Instant::now();
    let mut ctx = InferenceContext::from_gguf(model_path, BackendFactory::cpu())?;
    eprintln!("Loaded in {:.2}s", start.elapsed().as_secs_f64());

    eprintln!(
        "Config: arch={}, layers={}, embd={}, ff={}, heads={}",
        ctx.config().arch,
        ctx.config().block_count,
        ctx.config().embedding_length,
        ctx.config().feed_forward_length,
        ctx.config().attention.head_count,
    );
    eprintln!("Vocab size: {}", ctx.tokenizer().vocab_size());

    // Check weight shapes
    if let Some(w) = ctx.weights().get("token_embd.weight") {
        eprintln!("token_embd.weight: shape={:?}, {} values", w.shape, w.len());
    }
    if let Some(w) = ctx.weights().get("blk.0.ffn_gate.weight") {
        eprintln!("ffn_gate.weight: shape={:?}, {} values", w.shape, w.len());
    }
    if let Some(w) = ctx.weights().get("blk.0.attn_q.weight") {
        eprintln!("attn_q.weight: shape={:?}, {} values", w.shape, w.len());
    }

    // Forward pass with a single token (token 0 = <unk> or <s>)
    let tokens: Vec<u32> = vec![1]; // try token 1 instead of 0

    let start = Instant::now();
    let logits = ctx.forward(&tokens)?;
    eprintln!("Forward pass: {:.2}s", start.elapsed().as_secs_f64());

    // Show the logits distribution
    eprintln!("Logits: {} values", logits.len());

    // Stats
    let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = logits.iter().sum::<f32>() / logits.len() as f32;
    let zeros = logits.iter().filter(|v| **v == 0.0).count();
    let finites = logits.iter().filter(|v| v.is_finite()).count();

    eprintln!("  min: {min}");
    eprintln!("  max: {max}");
    eprintln!("  mean: {mean}");
    eprintln!("  zeros: {zeros}/{}", logits.len());
    eprintln!("  finite: {finites}/{}", logits.len());

    // Top-10 tokens
    let mut indexed: Vec<(f32, usize)> = logits.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    eprintln!("  Top-10 logits:");
    for (i, (val, idx)) in indexed.iter().take(10).enumerate() {
        eprintln!("    {}: token {idx} = {val:.6}", i + 1);
    }

    // Bottom-5
    eprintln!("  Bottom-5 logits:");
    for (i, (val, idx)) in indexed.iter().rev().take(5).enumerate() {
        eprintln!("    {}: token {idx} = {val:.6}", i + 1);
    }

    // Check embedding layer output
    eprintln!("\n=== Embedding check ===");
    let token_id = tokens[0] as usize;
    let n_embd = 2048;
    // The embedding is the first layer of the model
    // Let's check if embedding weights look reasonable
    if let Some(emb_w) = ctx.weights().get("token_embd.weight") {
        eprintln!(
            "  Embedding tensor: {:?}, {} values",
            emb_w.shape,
            emb_w.len()
        );
        let start = token_id * n_embd;
        let raw_bytes = emb_w.raw().len();
        // embedding_row_to_f32 dequants on demand; check bounds via raw byte count
        eprintln!("  Token {token_id} embedding start offset={start}, raw bytes={raw_bytes}");
        if let Ok(emb_slice) = emb_w.embedding_row_to_f32(token_id) {
            let emb_min = emb_slice.iter().copied().fold(f32::INFINITY, f32::min);
            let emb_max = emb_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let emb_mean: f32 = emb_slice.iter().sum::<f32>() / n_embd as f32;
            eprintln!("  Token {token_id} embedding: min={emb_min:.4}, max={emb_max:.4}, mean={emb_mean:.4}");
            eprintln!("  First 10 values: {:?}", &emb_slice[..10.min(emb_slice.len())]);
        }
    }

    Ok(())
}
