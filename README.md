# PowerInfer_x64

[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-stable-blue.svg)](https://www.rust-lang.org)
[![CUDA](https://img.shields.io/badge/CUDA-optional-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Vulkan](https://img.shields.io/badge/Vulkan-optional-7a6ff0.svg)](https://www.khronos.org/vulkan/)

Pure Rust prototype for PowerInfer-style neuron-sparse LLM inference. The long-term goal is to run modern hybrid and MoE models such as Qwen3.5-35B-A3B on constrained GPUs by keeping hot neurons on GPU and cold weights in CPU RAM.

## Status Snapshot

As of 2026-04-02, this repository is a working prototype with verified CPU inference and verified real-model output, not yet a finished sparse GPU runtime.

- `cargo test` passes: 81 tests passed, 2 ignored path-dependent quality checks, 0 failed.
- `cargo clippy --all-targets -- -D warnings` passes.
- `cargo run --release --bin gguf_dump -- /path/to/model.gguf` works and correctly reports metadata.
- `cargo run --release --bin real_test -- /path/to/model.gguf` works and produces a correct first token on the flagship validation model, Qwen3.5-35B-A3B Q8_0.
- The latest verified release-mode decode result on the development box is `1.44-1.46 tok/s` average on the CPU backend.
- CUDA and Vulkan compilation, hardware detection, and execution planning exist, but the planned GPU split is not yet dispatched end to end.
- The HTTP server now generates real model-backed completions, but it is still partial overall:
  - `/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/health`, and `/metrics` are live against the loaded model;
  - generation is currently greedy-only, non-streaming, and serialized behind a single model lock;
  - the profiler CLI still only inspects model structure and exits with a clear unsupported error instead of writing a profile.

## Performance Target

These goal numbers are intentionally unchanged. They are the destination for the repository, not the current measured throughput.

| Model | Hardware | VRAM | Target tok/s |
|-------|----------|------|--------------|
| Qwen3.5-35B-A3B Q4 | 2× GTX 1050 Ti | 7.5GB | 2.5–4 |
| Qwen3-8B Q4 | 2× GTX 1050 Ti | 5GB | 12–16 |
| Llama2-7B Q4 | 2× GTX 1050 Ti | 4.5GB | 15–20 |
| Qwen3-8B Q4 | Jetson Orin Nano | 6GB shared | 4–6 |

## Current Measured Result

Latest verified benchmark path:

```bash
cargo run --release --bin real_test -- /path/to/model.gguf
```

Current flagship measurement:

| Item | Value |
|------|-------|
| Validation model | Qwen3.5-35B-A3B Q8_0 |
| Model facts | 34.37 GiB, 733 tensors, 40 layers, 256 experts, 8 active experts |
| Development hardware | Pentium G4400, 32 GB DDR4, SSD, 2× GTX 1050 Ti present |
| Backend in actual run | CPU |
| GGUF parse | 1.85s |
| Inference context build | 3.57s |
| Prefill | 3.39s |
| Decode average | 1.44-1.46 tok/s across repeated release runs |
| Best observed token | 1.71 tok/s |
| Output | `Paris.ĊChooseĠtheĠcorrectĠanswerĠbelow:Ġmassive` |
| First token check | `Paris` ✅ |

All speed claims in this repository should be read as `--release` numbers. The same real-model test in the dev profile was about `0.02 tok/s`, which is useful for debugging but not for performance reporting.

## What Works Today

- GGUF parsing for `qwen2`, `qwen3`, `llama`, and `qwen35moe` metadata layouts.
- Memory-mapped weight loading, including zero-copy access for large expert tensors.
- CPU inference through the current Qwen3.5 validation path.
- Real-model generation with a correct first token on a production-size GGUF.
- Quantized CPU kernels and SIMD paths used by the current inference implementation.
- Hardware scanning and execution-plan generation for CPU + dual 1050 Ti systems.
- TurboQuant KV cache integration in the forward path.
- Prometheus metrics plumbing and HTTP routing scaffolding.

## What Is Still Incomplete

- GPU execution is planned but not yet wired through the runtime for end-to-end token generation.
- The core PowerInfer goal, sparse hot-neuron GPU execution with cold-neuron CPU fallback, is not implemented yet.
- The server does not yet support streaming or sampling controls; the validated path is greedy generation only.
- The predictor and profiler paths are scaffolding, not a finished hot-index pipeline.
- Sampling is still limited; greedy decoding is the current tested path.
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
| TurboQuant KV cache wiring | Active | Enabled in the execution plan for the current runtime path |

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
| Profiler to hot-index pipeline | Partial |
| OpenAI-compatible inference server | Partial, model-backed |
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
    --prompt "The capital of France is" \
    -n 1
```

### Start The HTTP Server

```bash
cargo run --release --features server --bin powerinfer-serve -- /path/to/model.gguf
```

Current limitation: the server now returns real model output, but only for non-streaming greedy generation. Sampling controls and fuller protocol compatibility are still incomplete.

### Run The Profiler Scaffold

```bash
cargo run --release --features profiling --bin powerinfer-profile -- \
    --model /path/to/model.gguf \
    --output profile.jsonl
```

Current limitation: this path currently performs model analysis only, then exits with a clear unsupported error. It does not produce a finished hot-neuron profile or index.

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
