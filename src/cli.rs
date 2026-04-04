//! PowerInfer_x64 command-line interface
//!
//! Entry point for `powerinfer-cli`, `powerinfer-profile`, and `powerinfer-serve`

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Resolve model path from CLI arg, env var, or error.
fn resolve_model(model: &Option<PathBuf>) -> anyhow::Result<PathBuf> {
    if let Some(m) = model {
        return Ok(m.clone());
    }
    if let Ok(env_model) = std::env::var("POWERINFER_MODEL") {
        let p = PathBuf::from(&env_model);
        if p.exists() {
            return Ok(p);
        }
        anyhow::bail!("POWERINFER_MODEL={env_model} does not exist");
    }
    anyhow::bail!(
        "No model specified. Use --model <path> or set POWERINFER_MODEL=/path/to/model.gguf"
    );
}

#[derive(Parser)]
#[command(name = "powerinfer")]
#[command(about = "Neuron-level sparse LLM inference in pure Rust")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Easiest path: auto-detect hardware and run with safe defaults
    Easy {
        /// Path to GGUF model file (or set POWERINFER_MODEL env var)
        #[arg(short, long)]
        model: Option<PathBuf>,

        /// Prompt text (uses a built-in prompt if omitted)
        #[arg(short, long)]
        prompt: Option<String>,

        /// Number of tokens to generate
        #[arg(short, long, default_value = "64")]
        n: usize,

        /// Temperature for sampling
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,

        /// Nucleus sampling cutoff
        #[arg(long, default_value_t = 0.95)]
        top_p: f32,

        /// Repetition penalty (>1.0 to penalize repeats)
        #[arg(long, default_value_t = 1.1)]
        repetition_penalty: f32,

        /// Force CPU-only execution
        #[arg(long, default_value_t = false)]
        cpu_only: bool,

        /// Interactive mode: keep prompting after each generation
        #[arg(short, long, default_value_t = false)]
        interactive: bool,
    },

    /// Generate text completion
    Generate {
        /// Path to GGUF model file (or set POWERINFER_MODEL env var)
        #[arg(short, long)]
        model: Option<PathBuf>,

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

        /// Nucleus sampling cutoff
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,

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
        #[arg(long, default_value_t = 0)]
        layers: usize,
        #[arg(long, default_value_t = 0.5)]
        threshold: f32,
        #[arg(long, default_value_t = 0.05)]
        min_hotness: f64,
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

/// Wrap a user prompt with Qwen3.5 / ChatML-style chat template.
/// This is needed because instruct-tuned models expect structured input.
fn apply_chat_template(user_message: &str) -> String {
    format!("<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n")
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Easy {
            ref model,
            ref prompt,
            n,
            temperature,
            top_p,
            repetition_penalty,
            cpu_only,
            interactive,
        } => {
            let model = resolve_model(model)?;
            let has_cuda_build = cfg!(feature = "cuda");

            eprintln!("Auto-detecting hardware...");
            let hw = powerinfer::sysinfo::HardwareProfile::sweep();
            let n_gpus = hw.gpus.len();

            eprintln!("Model: {}", model.display());
            eprintln!(
                "Detected: {} CPU cores, {} GPU(s), {:.1} GB RAM available",
                hw.cpu.logical_cores,
                n_gpus,
                hw.available_ram as f64 / (1024.0 * 1024.0 * 1024.0)
            );

            if cpu_only {
                eprintln!("Mode: CPU-only (forced by --cpu-only)");
            } else if has_cuda_build && n_gpus > 0 {
                eprintln!("Mode: CUDA GPU offload ({n_gpus} GPU(s) — hot layers on GPU, cold on CPU)");
            } else if n_gpus > 0 && !has_cuda_build {
                eprintln!("╔═══════════════════════════════════════════════════════════╗");
                eprintln!("║  ⚠  {n_gpus} NVIDIA GPU(s) detected but CUDA not enabled!    ║");
                eprintln!("║  Rebuild with: cargo build --release --features cuda      ║");
                eprintln!("║  Running CPU-only — expect 2-4× slower inference.         ║");
                eprintln!("╚═══════════════════════════════════════════════════════════╝");
            } else {
                eprintln!("Mode: CPU (no NVIDIA GPUs detected)");
            }

            // The backend object is always CPU — GPU offloading happens
            // internally in InferenceContext when compiled with --features cuda.
            let backend = powerinfer::runtime::BackendFactory::cpu();
            let mut ctx = powerinfer::model::InferenceContext::from_gguf(&model, backend)?;

            if interactive {
                eprintln!();
                eprintln!("Interactive mode. Type your prompt, press Enter to generate.");
                eprintln!("Type 'quit' or Ctrl-D to exit.");
                eprintln!();
                loop {
                    eprint!("You> ");
                    let mut line = String::new();
                    if std::io::stdin().read_line(&mut line)? == 0 {
                        break; // EOF
                    }
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    if line == "quit" || line == "exit" {
                        break;
                    }
                    let opts = powerinfer::GenerationOptions {
                        max_tokens: n,
                        temperature,
                        top_p,
                        repetition_penalty,
                        ..powerinfer::GenerationOptions::default()
                    };
                    let templated = apply_chat_template(line);
                    eprint!("AI> ");
                    let (output, token_times) = ctx.generate_streaming(&templated, opts, |token_text| {
                        eprint!("{token_text}");
                    })?;
                    let _ = output; // already printed via streaming
                    eprintln!();
                    if token_times.len() > 1 {
                        let decode_times = &token_times[1..];
                        let avg_s = decode_times.iter().sum::<f64>() / decode_times.len() as f64;
                        let tok_s = 1.0 / avg_s;
                        eprintln!("[{tok_s:.2} tok/s, {} tokens, {:.0}ms avg]", decode_times.len(), avg_s * 1000.0);
                    }
                    eprintln!();
                }
            } else {
                let raw_prompt = prompt
                    .clone()
                    .unwrap_or_else(|| "Please provide a successful list of 20 prime numbers.".to_string());
                let user_prompt = apply_chat_template(&raw_prompt);

                let opts = powerinfer::GenerationOptions {
                    max_tokens: n,
                    temperature,
                    top_p,
                    repetition_penalty,
                    ..powerinfer::GenerationOptions::default()
                };
                let (output, token_times) = ctx.generate_streaming(&user_prompt, opts, |token_text| {
                    print!("{token_text}");
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                })?;
                let _ = output;
                println!();

                if token_times.len() > 1 {
                    let decode_times = &token_times[1..];
                    let avg_s = decode_times.iter().sum::<f64>() / decode_times.len() as f64;
                    let tok_s = 1.0 / avg_s;
                    let prefill_ms = token_times[0] * 1000.0;
                    eprintln!();
                    eprintln!("Performance: {tok_s:.2} tok/s | {} tokens | {:.0}ms avg decode | {:.0}ms prefill", decode_times.len(), avg_s * 1000.0, prefill_ms);
                }
            }
        }
        Commands::Generate {
            ref model,
            ref prompt,
            n,
            gpu_layers,
            ref hot_index,
            temperature,
            top_p,
            _threads: _,
        } => {
            let model = resolve_model(model)?;
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

                let mut ctx = powerinfer::model::InferenceContext::from_gguf(&model, backend)?;
                if let Some(hot_index_path) = hot_index {
                    let index = powerinfer::activation::HotNeuronIndex::load(hot_index_path)?;
                    let indexed_layers = index.layers.len();
                    let indexed_units: usize =
                        index.layers.iter().map(|layer| layer.hot_indices.len()).sum();
                    ctx.set_hot_index(index)?;
                    eprintln!(
                        "Loaded hot index: {} layers, {} tracked units ({})",
                        indexed_layers,
                        indexed_units,
                        hot_index_path.display()
                    );
                }
                let output = ctx.generate_with_options(
                    prompt,
                    powerinfer::GenerationOptions {
                        max_tokens: n,
                        temperature,
                        top_p,
                        ..powerinfer::GenerationOptions::default()
                    },
                )?;
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
            layers,
            threshold,
            min_hotness,
        } => {
            eprintln!("Profiling model: {model:?}");
            eprintln!("Output: {output:?}");
            eprintln!("Prompts: {prompts:?}");
            eprintln!("Samples: {samples}");
            let mut prompt_texts = powerinfer::activation::load_prompts_from_files(prompts)?;
            if prompt_texts.is_empty() {
                prompt_texts = powerinfer::activation::default_profile_prompts();
                eprintln!("No prompt files provided; using built-in smoke prompts.");
            }

            let mut ctx = powerinfer::model::InferenceContext::from_gguf(
                model,
                powerinfer::runtime::BackendFactory::cpu(),
            )?;
            let result = powerinfer::activation::run_activation_profiling(
                &mut ctx,
                &prompt_texts,
                (layers > 0).then_some(layers),
                threshold,
                min_hotness,
                samples,
            )?;
            result.hot_index.save(output)?;

            eprintln!("{}", result.profile.summary());
            eprintln!("Prompts processed: {}", result.prompts_processed);
            eprintln!("Layers profiled: {}", result.hot_index.layers.len());
            if ctx.config().moe.is_some() {
                eprintln!(
                    "MoE note: profiled indices currently reflect router expert hotness on MoE layers; dense FFN layers still record neuron activations."
                );
            }
            eprintln!("Saved hot index to {output:?}");
        }
        Commands::Serve {
            ref model,
            port,
            concurrency,
            ref hot_index,
        } => {
            #[cfg(not(feature = "server"))]
            {
                let _ = (model, port, concurrency, hot_index);
                eprintln!("Error: server feature not enabled. Rebuild with --features server");
                std::process::exit(1);
            }
            #[cfg(feature = "server")]
            {
                eprintln!("Starting server on port {port}");
                eprintln!("Model: {model:?}");
                eprintln!("Concurrency: {concurrency}");
                if let Some(hot_index) = hot_index {
                    eprintln!("Hot index: {}", hot_index.display());
                }
                use powerinfer::server::{serve, ServerConfig};
                let config = ServerConfig {
                    host: "0.0.0.0".to_string(),
                    port,
                    model_path: model.to_string_lossy().into_owned(),
                    max_concurrent: concurrency,
                    max_queue_depth: 64,
                    hot_index_path: hot_index
                        .as_ref()
                        .map(|path| path.to_string_lossy().into_owned()),
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
