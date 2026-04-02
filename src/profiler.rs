//! Activation profiling CLI tool
//!
//! This binary profiles model activations to build hot neuron indices.

use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use powerinfer::activation::{
    default_profile_prompts,
    load_prompts_from_files,
    run_activation_profiling,
};
use powerinfer::runtime::BackendFactory;

#[derive(Parser)]
#[command(name = "powerinfer-profile")]
#[command(about = "Profile model activations to build hot neuron index")]
struct Cli {
    /// Path to GGUF model file
    #[arg(short, long)]
    model: PathBuf,

    /// Output file path (JSON hot-index format)
    #[arg(short, long)]
    output: PathBuf,

    /// Newline-delimited prompt file(s) to profile.
    #[arg(long = "prompt-file")]
    prompt_files: Vec<PathBuf>,

    /// Inline prompt(s) to include in the profiling run.
    #[arg(long = "prompt")]
    prompts: Vec<String>,

    /// Maximum number of prompts to process.
    #[arg(long, default_value_t = 64)]
    samples: usize,

    /// Number of layers to profile (0 = all layers)
    #[arg(long, default_value_t = 0)]
    layers: usize,

    /// Magnitude threshold used to mark a unit as hot within a sample.
    #[arg(long, default_value_t = 0.5)]
    threshold: f32,

    /// Minimum hotness ratio required for export into the hot index.
    #[arg(long, default_value_t = 0.05)]
    min_hotness: f64,
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
    let mut prompts = load_prompts_from_files(&cli.prompt_files)
        .with_context(|| "failed to load prompt files for activation profiling")?;
    prompts.extend(cli.prompts.clone());
    if prompts.is_empty() {
        prompts = default_profile_prompts();
        eprintln!("No prompt sources provided; using built-in smoke prompts.");
    }

    let mut ctx = powerinfer::model::InferenceContext::from_gguf(
        &cli.model,
        BackendFactory::cpu(),
    )?;

    eprintln!("Model loaded in {:.1}s", start.elapsed().as_secs_f32());
    eprintln!("Architecture: {}", ctx.config().arch);
    eprintln!("Layers: {}", ctx.config().block_count);
    eprintln!("Embedding dim: {}", ctx.config().embedding_length);
    eprintln!("FFN dim: {}", ctx.config().feed_forward_length);
    eprintln!();

    let result = run_activation_profiling(
        &mut ctx,
        &prompts,
        (cli.layers > 0).then_some(cli.layers),
        cli.threshold,
        cli.min_hotness,
        cli.samples,
    )?;
    result.hot_index.save(&cli.output)?;

    eprintln!("{}", result.profile.summary());
    eprintln!("Prompts processed: {}", result.prompts_processed);
    eprintln!("Layers profiled: {}", result.hot_index.layers.len());
    if ctx.config().moe.is_some() {
        eprintln!(
            "MoE note: profiled indices currently reflect router expert hotness on MoE layers; dense FFN layers still record neuron activations."
        );
    }
    eprintln!("Saved hot index to {}", cli.output.display());

    Ok(())
}
