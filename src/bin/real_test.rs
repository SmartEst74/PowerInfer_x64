//! Real model integration test binary.
//!
//! Loads a GGUF model, runs forward pass, and reports results.
//! Usage: cargo run --bin real_test -- <path_to.gguf> [--sort-bench]

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use powerinfer::gguf::GgufFile;
use powerinfer::model::{GenerationOptions, InferenceContext};
use powerinfer::runtime::BackendFactory;
use powerinfer::sysinfo::MemoryGuard;

const SORT_BENCH_PROMPT: &str = "user: Write only this Rust function: pub fn sort_numbers(values: &mut [u16]) { /* implementation */ }. Make it your fastest sorter for about 1000 random values in 0..=1023. No prose. No markdown. No fn main. No external crates.\nassistant:";
const SORT_BENCH_MAX_TOKENS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TestMode {
    Smoke,
    SortBench,
}

#[derive(Debug)]
struct HarnessReport {
    cases_tested: usize,
    generated_s: f64,
    std_s: f64,
    counting_s: f64,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let mut args = env::args();
    let bin = args.next().unwrap_or_else(|| "real_test".to_string());
    let mut model_path = None;
    let mut mode = TestMode::Smoke;

    for arg in args {
        match arg.as_str() {
            "--sort-bench" => mode = TestMode::SortBench,
            other if model_path.is_none() => model_path = Some(other.to_string()),
            _ => {
                eprintln!("Usage: {bin} <path_to.gguf> [--sort-bench]");
                std::process::exit(2);
            }
        }
    }

    let model_path = model_path.unwrap_or_else(|| {
        eprintln!("Usage: {bin} <path_to.gguf> [--sort-bench]");
        std::process::exit(2);
    });

    println!("=== PowerInfer_x64 Real Model Test ===");
    println!("Model: {model_path}");
    println!(
        "Mode: {}",
        match mode {
            TestMode::Smoke => "smoke prompt",
            TestMode::SortBench => "Rust codegen sort benchmark",
        }
    );
    println!();

    // Step 0: Memory preflight check
    let model_file_bytes = std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);
    let guard = MemoryGuard::check();
    guard.preflight(model_file_bytes);
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
    match mode {
        TestMode::Smoke => run_smoke_prompt(&mut ctx)?,
        TestMode::SortBench => run_sort_benchmark(&mut ctx)?,
    }

    Ok(())
}

fn run_smoke_prompt(ctx: &mut InferenceContext) -> anyhow::Result<()> {
    let prompt = "The capital of France is";
    println!("  Prompt: \"{prompt}\"");

    let start = Instant::now();
    // Run just the forward pass once to inspect logits
    let input_ids = ctx.tokenizer().encode(prompt);
    eprintln!(
        "  Token IDs for prompt: {:?}",
        &input_ids[..input_ids.len().min(10)]
    );
    eprintln!("  EOS token ID: {:?}", ctx.tokenizer().eos_token_id());

    let n_gen = 10;
    match ctx.generate_timed(prompt, n_gen) {
        Ok((output, token_times)) => {
            let elapsed = start.elapsed().as_secs_f64();
            println!("  Output: \"{output}\"");
            println!("  Total time: {elapsed:.2}s");

            // Print per-token timing
            if !token_times.is_empty() {
                let prefill_ms = token_times[0] * 1000.0;
                println!("  Prefill: {prefill_ms:.1}ms");
                if token_times.len() > 1 {
                    let decode_times: Vec<f64> = token_times[1..].to_vec();
                    let avg_ms =
                        decode_times.iter().sum::<f64>() / decode_times.len() as f64 * 1000.0;
                    let tok_s =
                        1.0 / (decode_times.iter().sum::<f64>() / decode_times.len() as f64);
                    println!("  Decode: {:.1}ms avg ({tok_s:.2} tok/s)", avg_ms);
                    for (i, &t) in decode_times.iter().enumerate() {
                        println!(
                            "    token {}: {:.1}ms ({:.2} tok/s)",
                            i + 1,
                            t * 1000.0,
                            1.0 / t
                        );
                    }
                }
            }

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

fn run_sort_benchmark(ctx: &mut InferenceContext) -> anyhow::Result<()> {
    std::env::set_var("POWERINFER_TRACE_TOKENS", "0");
    println!("  Benchmark prompt: asking the model for its most efficient Rust sort for the target workload.");
    let prompt_tokens = ctx.tokenizer().encode(SORT_BENCH_PROMPT).len();
    let start = Instant::now();
    let (output, token_times) = ctx.generate_timed_with_options(
        SORT_BENCH_PROMPT,
        GenerationOptions {
            max_tokens: SORT_BENCH_MAX_TOKENS,
            temperature: 0.0,
            top_p: 1.0,
            ..GenerationOptions::default()
        },
    )?;
    let elapsed = start.elapsed().as_secs_f64();

    let prefill_s = token_times.first().copied().unwrap_or(0.0);
    let decode_s: f64 = token_times.iter().skip(1).sum();
    let generated_tokens = token_times.len().saturating_sub(1);
    let decode_tok_s = if decode_s > 0.0 {
        generated_tokens as f64 / decode_s
    } else {
        0.0
    };
    let avg_decode_ms = if generated_tokens > 0 {
        decode_s / generated_tokens as f64 * 1000.0
    } else {
        0.0
    };

    println!("  Prompt tokens: {prompt_tokens}");
    println!("  Generated tokens: {generated_tokens}");
    println!("  Total generation time: {elapsed:.2}s");
    println!("  Prefill: {:.1}ms", prefill_s * 1000.0);
    if generated_tokens > 0 {
        println!(
            "  Decode: {:.1}ms avg ({decode_tok_s:.2} tok/s)",
            avg_decode_ms,
        );
    }

    let code = normalize_generated_code(&extract_rust_code(&output));
    println!("  Generated code bytes: {}", code.len());
    println!("  Code preview:");
    for line in code.lines().take(16) {
        println!("    {line}");
    }
    if code.lines().count() > 16 {
        println!("    ...");
    }

    let harness_dir = make_harness_dir()?;
    let harness_path = harness_dir.join("sort_harness.rs");
    let binary_path = harness_dir.join("sort_harness");
    fs::write(&harness_path, build_sort_harness(&code))?;

    let compile = Command::new("rustc")
        .arg("--edition")
        .arg("2021")
        .arg("-O")
        .arg(&harness_path)
        .arg("-o")
        .arg(&binary_path)
        .output()?;

    if !compile.status.success() {
        println!("  Harness compile: FAILED");
        if !compile.stderr.is_empty() {
            println!(
                "  rustc stderr:\n{}",
                String::from_utf8_lossy(&compile.stderr)
            );
        }
        anyhow::bail!(
            "generated Rust did not compile; harness saved at {}",
            harness_path.display()
        );
    }

    println!("  Harness compile: ok");
    let run = Command::new(&binary_path).output()?;
    if !run.status.success() {
        println!("  Harness execution: FAILED");
        if !run.stdout.is_empty() {
            println!("  stdout:\n{}", String::from_utf8_lossy(&run.stdout));
        }
        if !run.stderr.is_empty() {
            println!("  stderr:\n{}", String::from_utf8_lossy(&run.stderr));
        }
        anyhow::bail!(
            "generated sorter failed correctness harness; harness saved at {}",
            harness_path.display()
        );
    }

    let stdout = String::from_utf8_lossy(&run.stdout);
    let report = parse_harness_report(&stdout)?;
    let (best_label, best_baseline) = if report.std_s <= report.counting_s {
        ("std::sort_unstable baseline", report.std_s)
    } else {
        ("counting-sort baseline", report.counting_s)
    };
    let ratio_to_best = if best_baseline > 0.0 {
        report.generated_s / best_baseline
    } else {
        0.0
    };

    println!(
        "  Harness correctness: pass ({} cases)",
        report.cases_tested
    );
    println!("  Generated sorter: {:.4}s", report.generated_s);
    println!("  std::sort_unstable baseline: {:.4}s", report.std_s);
    println!("  counting-sort baseline: {:.4}s", report.counting_s);
    println!("  Best baseline: {best_label}");
    println!("  Ratio to best baseline: {:.2}x", ratio_to_best);
    println!("  Judgement: {}", performance_judgement(ratio_to_best));
    println!();
    println!("=== SUCCESS ===");
    println!("Model generated code, the harness compiled it, and correctness checks passed.");

    Ok(())
}

fn extract_rust_code(output: &str) -> String {
    let trimmed = output.trim();
    if let Some(fence_start) = trimmed.find("```") {
        let after_start = &trimmed[fence_start + 3..];
        let after_lang = after_start
            .strip_prefix("rust")
            .or_else(|| after_start.strip_prefix("Rust"))
            .unwrap_or(after_start);
        let after_newline = after_lang.strip_prefix('\n').unwrap_or(after_lang);
        if let Some(fence_end) = after_newline.find("```") {
            return after_newline[..fence_end].trim().to_string();
        }
    }
    trimmed.to_string()
}

fn normalize_generated_code(code: &str) -> String {
    let trimmed = code.trim();
    let Some(fn_start) = trimmed.find("pub fn sort_numbers") else {
        return trimmed.to_string();
    };

    let fn_code = &trimmed[fn_start..];
    let Some(open_brace) = fn_code.find('{') else {
        return fn_code.trim().to_string();
    };

    let mut depth = 0usize;
    for (idx, ch) in fn_code.char_indices().skip(open_brace) {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return fn_code[..=idx].trim().to_string();
                }
            }
            _ => {}
        }
    }

    fn_code.trim().to_string()
}

fn performance_judgement(ratio_to_best: f64) -> &'static str {
    if ratio_to_best <= 1.10 {
        "excellent: essentially tied with the fastest baseline"
    } else if ratio_to_best <= 1.50 {
        "good: correct and still close to the fastest baseline"
    } else if ratio_to_best <= 3.0 {
        "mixed: correct, but meaningfully slower than the fastest baseline"
    } else {
        "weak: correct, but far slower than the fastest baseline"
    }
}

fn make_harness_dir() -> anyhow::Result<PathBuf> {
    let ts = chrono::Utc::now()
        .timestamp_nanos_opt()
        .unwrap_or_else(|| chrono::Utc::now().timestamp_micros() * 1000);
    let dir = env::temp_dir().join(format!("powerinfer-sort-bench-{}-{ts}", std::process::id()));
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn build_sort_harness(generated_code: &str) -> String {
    format!(
        r#"{}

use std::hint::black_box;
use std::time::Instant;

fn next_u16(state: &mut u64) -> u16 {{
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 32) as u16) & 1023
}}

fn make_cases(seed: u64, cases: usize, len: usize) -> Vec<Vec<u16>> {{
    let mut state = seed;
    let mut out = Vec::with_capacity(cases);
    for _ in 0..cases {{
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {{
            values.push(next_u16(&mut state));
        }}
        out.push(values);
    }}
    out
}}

fn counting_sort(values: &mut [u16]) {{
    let mut counts = [0usize; 1024];
    for &value in values.iter() {{
        counts[value as usize] += 1;
    }}
    let mut write_idx = 0usize;
    for (value, &count) in counts.iter().enumerate() {{
        for _ in 0..count {{
            values[write_idx] = value as u16;
            write_idx += 1;
        }}
    }}
}}

fn verify(values: &[u16]) {{
    let mut actual = values.to_vec();
    let mut expected = values.to_vec();
    sort_numbers(&mut actual);
    expected.sort_unstable();
    assert_eq!(actual, expected);
}}

fn benchmark<F>(cases: &[Vec<u16>], sorter: F) -> f64
where
    F: Fn(&mut [u16]),
{{
    let start = Instant::now();
    for values in cases {{
        let mut working = values.clone();
        sorter(black_box(&mut working));
        black_box(&working);
    }}
    start.elapsed().as_secs_f64()
}}

fn main() {{
    let mut cases_tested = 0usize;

    let edge_cases = vec![
        vec![],
        vec![0],
        vec![1023],
        vec![5, 4, 3, 2, 1],
        vec![7; 1000],
        (0..1000).map(|i| (i % 1024) as u16).collect::<Vec<_>>(),
        (0..1000).rev().map(|i| (i % 1024) as u16).collect::<Vec<_>>(),
    ];

    for case in &edge_cases {{
        verify(case);
        cases_tested += 1;
    }}

    let random_cases = make_cases(0xC0FFEE, 64, 1000);
    for case in &random_cases {{
        verify(case);
        cases_tested += 1;
    }}

    let bench_cases = make_cases(0xBAD5EED, 96, 1000);
    let generated_s = benchmark(&bench_cases, |values| sort_numbers(values));
    let std_s = benchmark(&bench_cases, |values| values.sort_unstable());
    let counting_s = benchmark(&bench_cases, |values| counting_sort(values));

    println!("REPORT cases_tested={{}}", cases_tested);
    println!("REPORT generated_s={{:.6}}", generated_s);
    println!("REPORT std_s={{:.6}}", std_s);
    println!("REPORT counting_s={{:.6}}", counting_s);
}}
"#,
        generated_code
    )
}

fn parse_harness_report(stdout: &str) -> anyhow::Result<HarnessReport> {
    let mut cases_tested = None;
    let mut generated_s = None;
    let mut std_s = None;
    let mut counting_s = None;

    for line in stdout.lines() {
        let Some(rest) = line.strip_prefix("REPORT ") else {
            continue;
        };
        let Some((key, value)) = rest.split_once('=') else {
            continue;
        };

        match key.trim() {
            "cases_tested" => cases_tested = Some(value.trim().parse()?),
            "generated_s" => generated_s = Some(value.trim().parse()?),
            "std_s" => std_s = Some(value.trim().parse()?),
            "counting_s" => counting_s = Some(value.trim().parse()?),
            _ => {}
        }
    }

    Ok(HarnessReport {
        cases_tested: cases_tested
            .ok_or_else(|| anyhow::anyhow!("missing cases_tested in harness output"))?,
        generated_s: generated_s
            .ok_or_else(|| anyhow::anyhow!("missing generated_s in harness output"))?,
        std_s: std_s.ok_or_else(|| anyhow::anyhow!("missing std_s in harness output"))?,
        counting_s: counting_s
            .ok_or_else(|| anyhow::anyhow!("missing counting_s in harness output"))?,
    })
}
