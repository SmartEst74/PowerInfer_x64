//! PowerInfer_x64 command-line interface
//!
//! Entry point for `powerinfer-cli`, `powerinfer-profile`, and `powerinfer-serve`

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "powerinfer")]
#[command(about = "Neuron-level sparse LLM inference in pure Rust")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate text completion
    Generate {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: PathBuf,

        /// Prompt text
        #[arg(short, long)]
        prompt: Option<String>,

        /// Number of tokens to generate
        #[arg(short, long, default_value = "100")]
        n: usize,

        /// Number of GPU layers to offload (0 = CPU only)
        #[arg(long, default_value = "0")]
        gpu_layers: usize,

        /// Path to hot neuron index (optional)
        #[arg(long)]
        hot_index: Option<PathBuf>,

        /// Temperature for sampling
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,

        /// Threads for CPU parallelization
        #[arg(short, long, default_value_t = num_cpus::get())]
        threads: usize,
    },

    /// Profile activations to build hot neuron index
    Profile {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long)]
        prompts: Vec<PathBuf>,
        #[arg(long, default_value_t = 1000)]
        samples: usize,
    },

    /// Start OpenAI-compatible server
    Serve {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(long, default_value = "8080")]
        port: u16,
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
        #[arg(long)]
        hot_index: Option<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Generate { ref model, ref prompt, n, gpu_layers, ref hot_index, temperature, threads } => {
            if let Some(prompt) = prompt {
                eprintln!("Loading model: {:?}", model);
                // TODO: Implement actual generation
                let backend = if gpu_layers > 0 {
                    #[cfg(feature = "cuda")]
                    {
                        crate::runtime::BackendFactory::cuda(0)?
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        eprintln!("CUDA feature not enabled; using CPU");
                        crate::runtime::BackendFactory::cpu()
                    }
                } else {
                    crate::runtime::BackendFactory::cpu()
                };

                let mut ctx = crate::model::InferenceContext::from_gguf(model, backend)?;
                let output = ctx.generate(prompt, n)?;
                println!("{}", output);
            } else {
                eprintln!("Error: prompt is required");
                std::process::exit(1);
            }
        }
        Commands::Profile { ref model, ref output, ref prompts, samples } => {
            eprintln!("Profiling model: {:?}", model);
            eprintln!("Output: {:?}", output);
            eprintln!("Prompts: {:?}", prompts);
            eprintln!("Samples: {}", samples);
            // TODO: call profiler
            eprintln!("Profiler not yet implemented");
        }
        Commands::Serve { ref model, port, concurrency, ref hot_index } => {
            eprintln!("Starting server on port {}", port);
            eprintln!("Model: {:?}", model);
            eprintln!("Concurrency: {}", concurrency);
            if let Some(idx) = hot_index {
                eprintln!("Hot index: {:?}", idx);
            }
            // Call server module
            use crate::server::{ServerConfig, serve};
            let config = ServerConfig {
                host: "0.0.0.0".to_string(),
                port,
                model_path: model.to_string_lossy().into_owned(),
                max_concurrent: concurrency,
                max_queue_depth: 64,
            };
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(serve(config))?;
        }
    }

    Ok(())
}
