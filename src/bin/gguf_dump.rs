//! Dump GGUF metadata keys for debugging.
//! Usage: cargo run --bin gguf_dump -- <path_to.gguf>

use powerinfer::gguf::GgufFile;
use std::env;

fn main() -> anyhow::Result<()> {
    let mut args = env::args();
    let bin = args.next().unwrap_or_else(|| "gguf_dump".to_string());
    let model_path = args.next().unwrap_or_else(|| {
        eprintln!("Usage: {bin} <path_to.gguf>");
        std::process::exit(2);
    });

    println!("Loading: {model_path}");
    let gguf = GgufFile::open(&model_path)?;

    println!("\n=== Metadata Keys ===");
    for key in [
        "general.architecture",
        "general.name",
        "general.parameter_count",
        "qwen3.context_length",
        "llama.context_length",
        "qwen3.embedding_length",
        "llama.embedding_length",
        "qwen3.block_count",
        "llama.block_count",
        "qwen3.attention.head_count",
        "llama.attention.head_count",
        "qwen3.attention.head_count_kv",
        "llama.attention.head_count_kv",
        "qwen3.feed_forward_length",
        "llama.feed_forward_length",
        "qwen3.rope.dimension",
        "llama.rope.dimension",
        "qwen3.expert_count",
        "llama.expert_count",
        "qwen3.expert_used_count",
        "llama.expert_used_count",
        "qwen3.full_attention_interval",
    ] {
        if let Some(val) = gguf.metadata(key) {
            println!("  {key}: {val}");
        }
    }

    println!("\n=== All Keys (first 50) ===");
    // We can't iterate keys directly, so let's try common patterns
    let arch = gguf
        .metadata("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    println!("  Architecture: {arch}");

    // Try architecture-specific keys
    for suffix in [
        "context_length",
        "embedding_length",
        "block_count",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.key_length",
        "attention.value_length",
        "attention.layer_norm_rms_epsilon",
        "feed_forward_length",
        "expert_feed_forward_length",
        "rope.dimension",
        "rope.dimension_count",
        "rope.freq_base",
        "expert_count",
        "expert_used_count",
        "full_attention_interval",
        "ssm.inner_size",
        "ssm.state_size",
        "ssm.conv_kernel",
        "ssm.time_step_rank",
        "ssm.group_count",
    ] {
        let key = format!("{arch}.{suffix}");
        if let Some(val) = gguf.metadata(&key) {
            println!("  {key}: {val}");
        }
    }

    println!("\n=== Tensors (first 15) ===");
    for t in gguf.tensors().iter().take(15) {
        let shape: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
        println!("  {} [{}] kind={}", t.name, shape.join(", "), t.kind);
    }
    println!("  Total: {}", gguf.tensors().len());

    // Show per-layer weight patterns for a few layers
    println!("\n=== Layer Weight Patterns ===");
    let tensors = gguf.tensors();
    for layer in [0, 1, 4, 8, 12, 20, 39] {
        let prefix = format!("blk.{layer}.");
        let layer_tensors: Vec<_> = tensors.iter()
            .filter(|t| t.name.starts_with(&prefix))
            .collect();
        if layer_tensors.is_empty() { continue; }
        let names: Vec<String> = layer_tensors.iter()
            .map(|t| {
                let short = t.name.strip_prefix(&prefix).unwrap_or(&t.name);
                let shape: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
                format!("{short}[{}]k{}", shape.join(","), t.kind)
            })
            .collect();
        println!("  blk.{layer}: {}", names.join("  "));
    }

    Ok(())
}
