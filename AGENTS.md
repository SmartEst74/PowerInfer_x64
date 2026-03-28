# AGENTS.md — AI Coding Assistant Handoff Guide

## Project Overview

PowerInfer_x64 is a neuron-level sparse LLM inference engine in pure Rust. The goal is to run Qwen3.5-35B-A3B (35B total params, 3B active per token, MoE) on 2× GTX 1050 Ti (8GB VRAM) by splitting computation: hot neurons on GPU, cold neurons on CPU.

## Current State (2026-03-28)

**What works:**
- GGUF v3 parser: loads real models (tested with Arch-Agent-3B Q8_0, 3.1GB)
- Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, F32, F16 dequantization
- Forward pass: RMSNorm, RoPE, multi-head attention, SwiGLU FFN — functional but slow (no SIMD)
- BPE tokenizer: loads vocab from GGUF metadata
- TurboQuant KV cache compression: algorithm implemented, 8x compression ratio
- Activation profiler: tracks neuron hotness, exports hot neuron index
- Prometheus metrics endpoint (/metrics)
- CI/CD pipeline with matrix testing

**What doesn't work yet:**
- Q4_K_M/Q5_K dequantization (blocks Qwen3-4B and Qwen3.5-35B-A3B)
- SIMD matmul (forward pass too slow for 3B+ models)
- MoE routing (needed for 35B-A3B)
- Sparse GPU execution (hot neurons on GPU, cold on CPU)
- Quality validation against real models

**Test count:** 51 tests, all pass, 0 clippy warnings

## Build & Test

```bash
# Build
cargo build

# Run all tests
cargo test

# Run with server feature
cargo test --features server

# Clippy (must pass with -D warnings)
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features server -- -D warnings

# Test against real model (needs GGUF file)
cargo run --bin real_test -- /path/to/model.gguf

# Dump GGUF metadata
cargo run --bin gguf_dump -- /path/to/model.gguf
```

## Project Structure

```
src/
  lib.rs              — public API, module declarations
  gguf/mod.rs         — GGUF parser, model config extraction
  quant/mod.rs        — dequantization kernels (Q4_0, Q8_0, etc.)
  ops.rs              — tensor ops (RMSNorm, RoPE, attention, SwiGLU, matvec)
  model/mod.rs        — forward pass, InferenceContext, KV cache
  weights.rs          — weight loader (GGUF → f32)
  tokenizer.rs        — BPE tokenizer from GGUF vocab
  turboquant/mod.rs   — KV cache compression (Google TurboQuant, ICLR 2026)
  activation/mod.rs   — neuron hotness profiler, HotNeuronIndex
  benchmark/mod.rs    — perplexity measurement, reference comparison
  metrics.rs          — Prometheus metrics (9 metrics)
  server/mod.rs       — Axum HTTP server, OpenAI-compatible API
  runtime/mod.rs      — Backend trait, CPU/CUDA/Vulkan backends
  cli.rs              — CLI entry point
  profiler.rs         — profiler binary entry point
  bin/real_test.rs    — real model test binary
  bin/gguf_dump.rs    — GGUF metadata dump binary
  bin/powerinfer-serve.rs — server binary

tests/
  smoke.rs            — 6 smoke tests (pipeline produces finite output)
  soak.rs             — 7 soak tests (10K iterations, memory stability)
  integration.rs      — 5 integration tests (multi-module pipelines)

docs/
  architecture.md     — full architecture with citations
  build.md            — build instructions
  performance.md      — performance tuning guide
```

## Key Design Decisions

1. **Architecture-agnostic GGUF**: Uses `general.architecture` prefix (qwen2, qwen3, llama) instead of hardcoded `llama.` keys
2. **Weights stored as f32**: Dequantize on load for simplicity. Can be optimized later with fused dequant+matvec.
3. **TurboQuant for KV cache**: Compresses keys to 3-bit, values to 3-bit. Asymmetric estimator computes attention directly from compressed data.
4. **Activation profiling**: Records FFN gate activations, identifies hot neurons by hotness ratio threshold.

## Next Steps (Priority Order)

1. **Q4_K_M dequantization** (Issue #37) — blocks Qwen3-4B and Qwen3.5-35B-A3B
   - Q4_K_M uses superblock scaling: 256 values, 2 scales (d, m) per 128 values
   - Reference: llama.cpp ggml-quants.c Q4_K implementation
   
2. **Q5_K dequantization** (Issue #38) — same superblock pattern, 5-bit

3. **SIMD matmul** (Issue #40-45) — makes inference fast enough
   - Use std::simd (f32x8, f16x8)
   - Tiled matmul (64×64, 128×128 for L1/L2 cache)
   - Fused dequant+matmul (avoid full dequant buffer)

4. **MoE routing** (Issue #81) — needed for 35B-A3B
   - Router selects top-k experts per token
   - Only active experts computed on GPU

5. **Sparse GPU execution** (Issue #75) — the core innovation
   - Hot neuron weights on GPU, cold on CPU
   - Predictor (tiny MLP) decides hot/cold

## What NOT to Do

- Don't add features without an open GitHub issue
- Don't claim "works" without a real model test
- Don't write tests only on synthetic data — test against real GGUF files
- Don't hardcode `llama.` prefix — use architecture-agnostic `get_config()`
- Don't add dependencies without checking Cargo.toml first
- Don't use `crate::` in binary files — use `powerinfer::`

## How to Verify Changes

1. `cargo clippy --all-targets -- -D warnings` — must pass
2. `cargo test` — all 51+ tests must pass
3. `cargo run --bin real_test -- /home/jon/models/llama-cache/Arch-Agent-3B.Q8_0.gguf` — GGUF must load
4. Update QUALITY.md with results

## Available Models (on disk)

```
/home/jon/models/llama-cache/
  Arch-Agent-3B.Q8_0.gguf              3.1GB  Q8_0   Qwen2 arch, 36 layers, 2048 dim
  Qwen3-4B-Instruct-2507-Q4_K_M.gguf  2.4GB  Q4_K_M Qwen3 arch, 36 layers, 2560 dim
  gemma-3-4b-it-Q4_K_M.gguf           2.4GB  Q4_K_M Gemma arch
  BitAgent-Bounty-8B.Q4_K_M.gguf      4.6GB  Q4_K_M
  Salesforce.Llama-xLAM-2-8b-fc-r.Q4_K_M.gguf 4.6GB Q4_K_M
```

## References (verified, not hallucinated)

- PowerInfer: arxiv.org/abs/2312.12456
- PowerInfer-2: arxiv.org/abs/2406.06282
- TurboQuant: arxiv.org/abs/2504.19874 (ICLR 2026)
- Qwen3.5-35B-A3B: huggingface.co/Qwen/Qwen3.5-35B-A3B
- gguf-rs: crates.io/crates/gguf-rs
- llama.cpp: github.com/ggml-org/llama.cpp (reference for quant formats)

## Important Notes

- The stale `/home/jon/Cargo.toml` and `/home/jon/src/` interfere with `cargo check`. Hide them before building:
  ```bash
  mv /home/jon/Cargo.toml /home/jon/Cargo.toml.bak
  mv /home/jon/src /home/jon/src.bak
  cargo check
  mv /home/jon/Cargo.toml.bak /home/jon/Cargo.toml
  mv /home/jon/src.bak /home/jon/src
  ```
- Rust toolchain: nightly-2025-06-23 (for rust-gpu)
- The project uses `anyhow::Result` throughout, not `std::result::Result`
