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
        _hot_index: Option<PathBuf>,

        /// Temperature for sampling
        #[arg(long, default_value_t = 0.7)]
        _temperature: f32,

        /// Threads for CPU parallelization
        #[arg(short, long, default_value_t = num_cpus::get())]
        _threads: usize,
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
        _hot_index: Option<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            ref model,
            ref prompt,
            n,
            gpu_layers,
            _hot_index: _,
            _temperature: _,
            _threads: _,
        } => {
            if let Some(prompt) = prompt {
                eprintln!("Loading model: {model:?}");
                let backend = if gpu_layers > 0 {
                    #[cfg(feature = "cuda")]
                    {
                        powerinfer::runtime::BackendFactory::cuda(0)?
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        eprintln!("CUDA feature not enabled; using CPU");
                        powerinfer::runtime::BackendFactory::cpu()
                    }
                } else {
                    powerinfer::runtime::BackendFactory::cpu()
                };

                let mut ctx = powerinfer::model::InferenceContext::from_gguf(model, backend)?;
                let output = ctx.generate(prompt, n)?;
                println!("{output}");
            } else {
                eprintln!("Error: prompt is required");
                std::process::exit(1);
            }
        }
        Commands::Profile {
            ref model,
            ref output,
            ref prompts,
            samples,
        } => {
            eprintln!("Profiling model: {model:?}");
            eprintln!("Output: {output:?}");
            eprintln!("Prompts: {prompts:?}");
            eprintln!("Samples: {samples}");
            anyhow::bail!(
                "activation profiling is not wired into the forward pass yet; use the standalone profiler only for model inspection until hot-index generation is implemented"
            );
        }
        Commands::Serve {
            ref model,
            port,
            concurrency,
            _hot_index: _,
        } => {
            #[cfg(not(feature = "server"))]
            {
                let _ = (model, port, concurrency);
                eprintln!("Error: server feature not enabled. Rebuild with --features server");
                std::process::exit(1);
            }
            #[cfg(feature = "server")]
            {
                eprintln!("Starting server on port {port}");
                eprintln!("Model: {model:?}");
                eprintln!("Concurrency: {concurrency}");
                use powerinfer::server::{serve, ServerConfig};
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
    }

    Ok(())
}
