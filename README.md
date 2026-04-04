# PowerInfer_x64

[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-stable-blue.svg)](https://www.rust-lang.org)
[![CUDA](https://img.shields.io/badge/CUDA-optional-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Vulkan](https://img.shields.io/badge/Vulkan-optional-7a6ff0.svg)](https://www.khronos.org/vulkan/)

Pure Rust prototype for PowerInfer-style neuron-sparse LLM inference. The long-term goal is to run modern hybrid and MoE models such as Qwen3.5-35B-A3B on constrained GPUs by keeping hot neurons on GPU and cold weights in CPU RAM.

## Status Snapshot

As of 2026-04-03, this repository is a working prototype with verified real-model output and verified CUDA-accelerated token generation on dual GTX 1050 Ti, not yet a finished sparse GPU runtime.

- `cargo test` passes: 86 tests passed, 2 ignored path-dependent quality checks, 0 failed.
- `cargo clippy --all-targets -- -D warnings` passes.
- `cargo run --release --bin gguf_dump -- /path/to/model.gguf` works and correctly reports metadata.
- `cargo run --release --bin real_test -- /path/to/model.gguf` works and produces a correct first token on the flagship validation model, Qwen3.5-35B-A3B Q8_0.
- The latest verified release-mode decode result on the development box is `2.23 tok/s` average with `--features cuda` on Qwen3.5-35B-A3B Q8_0.
- CUDA compilation, hardware detection, execution planning, and projection/LM-head offload are active; full sparse hot-neuron GPU execution is still not implemented.
- The HTTP server now generates real model-backed completions, but it is still partial overall:
  - `/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/health`, and `/metrics` are live against the loaded model;
  - non-streaming temperature/top-p sampling is wired through the CLI and HTTP API;
  - request handling is still serialized behind a single model lock;
  - the profiler CLI now writes a real JSON hot-index file from prompt inputs, and the runtime can load that format for experimental dense-FFN sparsity and hot MoE expert caching on the CPU path; this is still not the finished sparse GPU pipeline.

## Performance Target

These goal numbers are intentionally unchanged. They are the destination for the repository, not the current measured throughput.

| Model | Hardware | VRAM | Target tok/s |
|-------|----------|------|--------------|
| Qwen3.5-35B-A3B Q4 | 2× GTX 1050 Ti | 7.5GB | 2.5–4 |
| Qwen3-8B Q4 | 2× GTX 1050 Ti | 5GB | 12–16 |
| Llama2-7B Q4 | 2× GTX 1050 Ti | 4.5GB | 15–20 |
| Qwen3-8B Q4 | Jetson Orin Nano | 6GB shared | 4–6 |

## Quickstart For Any Hardware

This section is the fastest path from zero to first successful large-model tokens.

### Fastest possible start (recommended)

If you want one command and minimal decisions:

```bash
cargo run --release --features cuda --bin powerinfer-cli -- easy --model /path/to/model.gguf
```

What this does automatically:
- detects your hardware
- prefers CUDA when available
- falls back to CPU if CUDA is unavailable or fails
- runs with safe generation defaults

CPU-only fallback command:

```bash
cargo run --release --bin powerinfer-cli -- easy --model /path/to/model.gguf --cpu-only
```

### 1) Pick your runtime path

- If you have NVIDIA GPUs and CUDA installed: use `--features cuda`.
- If you do not have CUDA ready: start CPU-only first, then add CUDA later.
- If your machine is memory-constrained: start with a smaller GGUF, prove the full pipeline, then scale up.

### 2) Verify your model file and architecture

```bash
cargo run --release --bin gguf_dump -- /path/to/model.gguf
```

Expected signs of success:
- metadata prints without error
- tensor list appears
- architecture key is recognized (`qwen35moe`, `qwen3`, `qwen2`, or `llama`)

### 3) Run first end-to-end inference

CPU-first path:

```bash
cargo run --release --bin real_test -- /path/to/model.gguf
```

CUDA path:

```bash
cargo run --release --features cuda --bin real_test -- /path/to/model.gguf
```

Expected signs of success:
- model loads
- output text is produced
- `=== SUCCESS ===` is printed

### 4) Turn on per-token diagnostics when tuning

```bash
POWERINFER_TRACE_TOKENS=1 cargo run --release --features cuda --bin real_test -- /path/to/model.gguf
```

Use diagnostics to identify whether your bottleneck is layer compute, LM head, or memory pressure.

### 5) Use release-mode only for speed claims

Debug builds are intentionally much slower and should not be used for throughput comparisons.

## Current Measured Result

Latest verified benchmark paths:

```bash
cargo run --release --bin real_test -- /path/to/model.gguf
cargo run --release --features cuda --bin real_test -- /path/to/model.gguf
```

Current flagship measurement:

| Item | Value |
|------|-------|
| Validation model | Qwen3.5-35B-A3B Q8_0 |
| Model facts | 34.37 GiB, 733 tensors, 40 layers, 256 experts, 8 active experts |
| Development hardware | Pentium G4400, 32 GB DDR4, SSD, 2× GTX 1050 Ti present |
| Backend in actual run | CUDA-enabled runtime (`--features cuda`) |
| GGUF parse | 1.85s |
| Inference context build | 3.57s |
| Prefill | 3.39s |
| Decode average | 2.23 tok/s (448.1ms/token) |
| Best observed token | 2.34 tok/s |
| Output | `located in which country?` then `France` appears in continuation |
| Target check | 2.0 tok/s target met ✅ |

All speed claims in this repository should be read as `--release` numbers.

## What Works Today

- GGUF parsing for `qwen2`, `qwen3`, `llama`, and `qwen35moe` metadata layouts.
- Memory-mapped weight loading, including zero-copy access for large expert tensors.
- CPU inference through the current Qwen3.5 validation path.
- Real-model generation with a correct first token on a production-size GGUF.
- Non-greedy temperature/top-p sampling through `powerinfer-cli generate` and the HTTP server.
- Quantized CPU kernels and SIMD paths used by the current inference implementation.
- Hardware scanning and execution-plan generation for CPU + dual 1050 Ti systems.
- TurboQuant KV cache integration in the forward path.
- Prometheus metrics plumbing and HTTP routing scaffolding.
- Basic activation profiling that exports a JSON hot-index file from prompt sets.
- Experimental runtime loading of hot-index JSON through `powerinfer-cli generate` and `powerinfer-serve`.

## What Is Still Incomplete

- The sparse hot-neuron GPU path (the core PowerInfer design target) is still not implemented.
- The core PowerInfer goal, sparse hot-neuron GPU execution with cold-neuron CPU fallback, is not implemented yet.
- The server does not yet support streaming and still serializes requests through one shared model lock.
- The profiler exports usable hot-index JSON, and the CPU runtime now consumes it experimentally for dense FFN sparsity and hot MoE expert caching; MoE layers still export expert hotness rather than per-expert neuron hotness, and the GPU sparse path is still missing.
- The predictor path is still scaffolding and does not train or consume real profiler output yet.
- Qwen3.5 chat templating and reference comparison against llama.cpp are still open tasks.

## Optimizations Applied So Far

| Optimization | Current state | Observed effect |
|--------------|---------------|-----------------|
| SSE4.1 Q8_0 matvec (`cvtepi8 -> cvtepi32 -> f32`) | Active | ~2x vs scalar |
| Parallel expert FFN (2 threads, 4 experts each) | Active | ~2x expert throughput |
| Parallel matvec for large outputs (`n_out >= 4096`) | Active | ~1.7x on QKV and LM head |
| Batch mmap prefetch for all 8 active experts | Active | Fewer page faults during decode |
| Zero-copy expert weight access (`Arc<Mmap>`) | Active | Avoids per-expert allocations |
| Hardware-adaptive execution planning | Active | Computes CPU/GPU layer split and memory plan at load time |
| TurboQuant KV cache wiring | Active | Enabled in the runtime path |
| TurboQuant precomputed query rotation | Active | Reduced attention-core cost from double-digit ms to low single-digit ms in traced runs |
| CUDA projection offload | Active | Per-layer projection matvecs run on GPU when uploaded |
| LM head split across GPUs | Active | Large reduction in LM head decode time |
| Persistent GPU scratch buffers | Active | Avoids per-call CUDA alloc/free overhead |

## Hardware Tuning Checklist

Use this checklist when adapting to a new machine.

- Keep model and binaries on fast local SSD.
- Run `cargo build --release` before benchmarking to avoid compile noise in timing.
- Prefer CUDA path when NVIDIA GPUs are present and stable.
- Do not benchmark with extra debug logging unless you are diagnosing a bottleneck.
- Keep other heavy processes off the same RAM/VRAM budget during tests.
- Compare at least 2 runs: first run (cold) and second run (warmer caches).
- Report exact command, model file, backend, and hardware in every benchmark note.

## Correctness Fixes Behind The Current Output

The current Qwen3.5 result only became credible after a set of architecture-specific fixes landed.

| Fix | Why it mattered |
|-----|-----------------|
| Added missing `1/sqrt(k_hd)` query scaling after L2 normalization | Restored correct attention magnitude |
| Fixed tiled V-head ordering and untiled `A_log` / `dt_bias` mapping | Removed garbled Gated Delta Rule behavior |
| Stopped double-exponentiating `ssm_a` | Restored correct state decay |
| Switched MoE routing from sigmoid to softmax -> top-k -> renormalize | Restored correct expert weighting |

## Progress Toward The Repo Goals

| Goal area | Current status |
|-----------|----------------|
| GGUF parsing for target architectures | Verified |
| Quantized CPU inference on real models | Verified |
| Correct first-token generation on Qwen3.5-35B-A3B | Verified |
| Release-mode throughput baseline on development hardware | Verified |
| CUDA and Vulkan compile path | Verified |
| Hardware-aware CPU/GPU planning | Verified |
| End-to-end GPU token generation | Partial |
| Sparse hot-neuron execution | Not done |
| Profiler to hot-index pipeline | Partial, export plus experimental CPU runtime consumption |
| OpenAI-compatible inference server | Partial, model-backed with basic sampling |
| Benchmark regression tracking in CI | Open issue |

## Journey So Far

- 2025-03-25: repository created with a 48-week plan focused on PowerInfer-style sparse execution.
- `6c4a5cd`: added quality validation tests against real model files.
- `fc1ed15`: first end-to-end inference produced output tokens.
- `4f29ce1`: fixed GGUF tensor data offset so weights load correctly.
- `d64758e`: moved the default toolchain to stable and added the CUDA backend skeleton.
- `f79d115`: wired TurboQuant compressed KV cache into the forward pass.
- `ddc5da8`: fixed four critical forward-pass bugs and added parallel expert computation.
- `ed20b06`: added hardware-adaptive optimizations and cross-platform detection.

## Build And Verify

### Toolchain

- Default toolchain: stable Rust from `rust-toolchain.toml`.
- Optional features: `cuda`, `vulkan`, `server`, and `profiling`.
- `nightly-2025-06-23` is only needed when experimenting with future `rust-gpu` kernel compilation.

### Core Verification Commands

```bash
# Build optimized binaries
cargo build --release

# Verify code health
cargo test
cargo clippy --all-targets -- -D warnings

# Inspect model metadata
cargo run --release --bin gguf_dump -- /path/to/model.gguf

# Run the real-model validation path
cargo run --release --bin real_test -- /path/to/model.gguf
```

### Generate Text

```bash
cargo run --release --bin powerinfer-cli -- generate \
    --model /path/to/model.gguf \
    --prompt "Please provide a successful list of 20 prime numbers." \
  -n 100 \
  --temperature 0.7 \
  --top-p 0.9
```

For most users, prefer `easy` over `generate`.

### Start The HTTP Server

```bash
cargo run --release --features server --bin powerinfer-serve -- /path/to/model.gguf
```

Pass a second path argument to load a hot-index file into the server runtime.

Current limitation: the server now returns real model output with basic temperature/top-p sampling, but it is still non-streaming and serialized through a single model lock.

### Run The Profiler

```bash
cargo run --release --features profiling --bin powerinfer-profile -- \
    --model /path/to/model.gguf \
  --output hot_index.json \
  --prompt-file prompts.txt
```

If no prompt file is provided, the profiler falls back to a small built-in smoke prompt set.

Current limitation: this path exports a real JSON hot index, and the CPU runtime can now load that format experimentally, but MoE layers still capture expert selection hotness rather than per-expert neuron hotness and the GPU sparse path is still not implemented.

## Infrastructure Notes

The repository includes substantial deployment and observability scaffolding under `deployments/` and `infrastructure/`, including Docker Compose, Prometheus, Grafana, Alertmanager, Terraform, and runbooks. Those assets are now aligned with a real-inference server path, but they should still be read as pre-production scaffolding rather than proof of a hardened deployment.

## Documentation Map

- `QUALITY.md`: current quality baseline, measured results, and gaps.
- `PROJECT_STATUS.md`: progress summary and current execution priorities.
- `PLAN.md`: 48-week implementation plan and long-term roadmap.
- `docs/build.md`: accurate build and run instructions for the current codebase.
- `docs/performance.md`: current benchmark reporting rules, bottlenecks, and tuning notes.
- `docs/architecture.md`: design intent for the eventual sparse runtime.

## License

Dual-licensed under MIT or Apache 2.0. See `LICENSE-MIT` and `LICENSE-APACHE` for details.
