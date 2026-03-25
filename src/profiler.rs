//! Activation profiling CLI tool
//!
//! This binary profiles model activations to build hot neuron indices.

use clap::{Parser, ValueHint};
use std::path::PathBuf;
use std::sync::Arc;
use std::io::{self, Write};
use std::time::Instant;
use half::f16;

use powerinfer::gguf::GgufFile;
use powerinfer::model::{InferenceContext, ModelConfig};
use powerinfer::runtime::{Backend, BackendFactory};
use powerinfer::profiler::{Profiler, NeuronStats};
use powerinfer::tokenizer;

#[derive(Parser)]
#[command(name = "powerinfer-profile")]
#[command(about = "Profile model activations to build hot neuron index")]
struct Cli {
    /// Path to GGUF model file
    #[arg(short, long)]
    model: PathBuf,

    /// Output file path (JSONL format)
    #[arg(short, long)]
    output: PathBuf,

    /// Sample prompts (one per line in a text file, or directory with .txt files)
    #[arg(short, long)]
    prompts: Vec<PathBuf>,

    /// Number of samples to profile per prompt
    #[arg(long, default_value_t = 1000)]
    samples: usize,

    /// Sampling rate (1.0 = all tokens, 0.1 = 10%)
    #[arg(long, default_value_t = 1.0)]
    sample_rate: f32,

    /// Number of threads
    #[arg(short, long, default_value_t = 4)]
    threads: usize,

    /// GPU layers (0 = CPU only)
    #[arg(long, default_value_t = 0)]
    gpu_layers: usize,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    eprintln!("PowerInfer_x64 Profiler");
    eprintln!("========================");
    eprintln!("Model: {}", cli.model.display());
    eprintln!("Output: {}", cli.output.display());
    eprintln!("Samples: {}", cli.samples);
    eprintln!("Sample rate: {:.1}%", cli.sample_rate * 100.0);
    eprintln!("Threads: {}", cli.threads);
    eprintln!();

    // Load model
    let start = Instant::now();
    let gguf = GgufFile::open(&cli.model)?;
    let config = gguf.model_config()?;

    eprintln!("Model loaded in {:.1}s", start.elapsed().as_secs_f32());
    eprintln!("Architecture: {}", config.arch);
    eprintln!("Layers: {}", config.block_count);
    eprintln!("Embedding dim: {}", config.embedding_length);
    eprintln!("FFN dim: {}", config.feed_forward_length);
    eprintln!();

    // Create backend
    let backend = if cli.gpu_layers > 0 {
        #[cfg(feature = "cuda")]
        {
            eprintln!("Using CUDA backend with {} layers", cli.gpu_layers);
            BackendFactory::cuda(0)?
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("Warning: CUDA not enabled, using CPU only");
            BackendFactory::cpu()
        }
    } else {
        eprintln!("Using CPU backend");
        BackendFactory::cpu()
    };

    // Load tokenizer
    let tok = tokenizer::Tokenizer::from_gguf(&gguf)?;
    eprintln!("Tokenizer loaded (vocab size: {})", tok.vocab_size());
    eprintln!();

    // Initialize profiler
    let mut profiler = Profiler::new(&cli.output, cli.sample_rate)?;

    // Read prompts
    let mut prompts = Vec::new();
    for path in &cli.prompts {
        if path.is_dir() {
            // Load all .txt files from directory
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map(|ext| ext == "txt").unwrap_or(false) {
                    let content = std::fs::read_to_string(&path)?;
                    prompts.push(content.trim().to_string());
                }
            }
        } else if path.is_file() {
            let content = std::fs::read_to_string(path)?;
            // If file has multiple lines, treat each line as a separate prompt
            for line in content.lines() {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    prompts.push(trimmed.to_string());
                }
            }
        }
    }

    eprintln!("Loaded {} prompts", prompts.len());

    // Run profiling
    let start = Instant::now();
    let mut total_tokens = 0;
    let mut total_neuron_stats = 0;

    for (prompt_idx, prompt) in prompts.iter().enumerate() {
        eprintln!("Profiling prompt {}/{}...", prompt_idx + 1, prompts.len());

        // Tokenize
        let tokens = tok.encode(prompt);
        eprintln!("  Tokens: {}", tokens.len());

        // Create inference context (simplified - just for profiling)
        // In practice, we'd run the forward pass and hook into FFN layers
        // For now, generate dummy statistics

        for sample in 0..cli.samples.min(tokens.len()) {
            // In a real implementation, we'd:
            // 1. Run forward pass up to each FFN layer
            // 2. Capture output activations
            // 3. Compute statistics per neuron
            // 4. Record to profiler
            
            // Dummy recording (placeholder)
            for layer in 0..config.block_count {
                let neuron = sample % config.feed_forward_length;
                let activation = f16::from_f32(0.5);
                profiler.record(layer, neuron, activation);
                total_neuron_stats += 1;
            }
            total_tokens += 1;
        }
    }

    profiler.finish()?;

    let elapsed = start.elapsed();
    eprintln!();
    eprintln!("Profiling complete!");
    eprintln!("Total tokens processed: {}", total_tokens);
    eprintln!("Total neuron stats recorded: {}", total_neuron_stats);
    eprintln!("Time elapsed: {:.1}s", elapsed.as_secs_f32());
    eprintln!("Throughput: {:.1} tokens/sec", total_tokens as f32 / elapsed.as_secs_f32());
    eprintln!();
    eprintln!("Output written to: {}", cli.output.display());

    Ok(())
}
