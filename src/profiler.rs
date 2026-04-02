//! Activation profiling CLI tool
//!
//! This binary profiles model activations to build hot neuron indices.

use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::bail;
use powerinfer::gguf::GgufFile;

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

    /// Number of layers to profile
    #[arg(long, default_value_t = 0)]
    layers: usize,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    eprintln!("PowerInfer_x64 Profiler");
    eprintln!("========================");
    eprintln!("Model: {}", cli.model.display());
    eprintln!("Output: {}", cli.output.display());
    eprintln!();

    let start = Instant::now();
    let gguf = GgufFile::open(&cli.model)?;
    let config = gguf.model_config()?;

    eprintln!("Model loaded in {:.1}s", start.elapsed().as_secs_f32());
    eprintln!("Architecture: {}", config.arch);
    eprintln!("Layers: {}", config.block_count);
    eprintln!("Embedding dim: {}", config.embedding_length);
    eprintln!("FFN dim: {}", config.feed_forward_length);
    eprintln!();

    bail!(
        "activation profiling is not wired into the forward pass yet; no profile was written to {}",
        cli.output.display()
    );

}
