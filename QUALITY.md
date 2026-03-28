# Quality Baseline

**Date**: 2026-03-28
**Commit**: HEAD on master
**Rust**: nightly-2025-06-23

## Test Results

| Category | Count | Status |
|----------|-------|--------|
| Unit tests | 32 | ✅ PASS |
| Smoke tests | 6 | ✅ PASS |
| Soak tests (10K iterations) | 7 | ✅ PASS |
| Integration tests | 5 | ✅ PASS |
| Doc tests | 1 | ✅ PASS |
| **Total** | **51** | **✅ ALL PASS** |

## Lint

| Check | Status |
|-------|--------|
| `cargo fmt --check` | ✅ PASS |
| `cargo clippy -D warnings` (no features) | ✅ PASS |
| `cargo clippy -D warnings` (--features server) | ✅ PASS |

## What Is Tested

### Unit Tests (src/**)
- Dequantization: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F32, F16 roundtrip
- Ops: RMSNorm, RoPE, softmax, SiLU, attention head, matvec, FFN SwiGLU
- TurboQuant: Lloyd-Max centroids, orthogonal matrix (Q^T Q = I), quantize/dequantize, asymmetric dot accuracy, bit packing, compression ratios
- Activation profiler: neuron stats, hot neurons, masks, export, summary, memory estimate
- Benchmark: log_softmax, argmax, topk

### Smoke Tests (tests/smoke.rs)
- Quantization produces finite output
- All ops produce finite output
- Matvec correctness (known values)
- TurboQuant compresses and restores
- Profiler records and exports correctly
- Benchmark math works

### Soak Tests (tests/soak.rs)
- RMSNorm: 10K iterations, output stays finite
- Softmax: 10K iterations, sum stays = 1.0
- Matvec: 10K iterations, output stays finite
- Dequantization: 10K iterations, output stays finite
- TurboQuant: 1K iterations, output stays finite
- Profiler: 10K iterations, no overflow
- Attention head: 100 iterations, output stays finite

### Integration Tests (tests/integration.rs)
- Profiler → hot index → memory estimation pipeline
- Compressed KV cache → attention scores → weighted value sum
- Reference comparison (identical vectors → zero error, cosine sim = 1)
- GPU memory planning for Llama-7B-sized model
- Full transformer forward pass (RMSNorm → Q projection → attention → FFN → residual)

## What Is NOT Yet Tested

| Requirement | Issue | Status |
|-------------|-------|--------|
| Forward pass vs llama.cpp reference (1e-4) | #31 | ❌ NOT DONE |
| Generate 512 tokens coherence check | #32 | ❌ NOT DONE |
| Qwen3-8B perplexity validation | #60 | ❌ NOT DONE |
| Benchmark: >10 tok/s on CPU | #33 | ❌ NOT DONE |
| Integration test with real GGUF file | — | ❌ NOT DONE |

These require a real GGUF model file and reference implementation comparison.

## CI Pipeline

- Lint: `cargo fmt` + `cargo clippy -D warnings` (both default and server features)
- Test: matrix strategy (no features + --features server)
- Security: `cargo audit`
- Benchmark: runs on push, stores artifacts

## Commit History (this session)

- `790d9e9` — Add proper citations to architecture doc
- `3b5fc55` — Add QA pipeline and TurboQuant sections to architecture doc
- `c844d99` — Add activation profiler and quality benchmark infrastructure
- `e0a8ed6` — Implement TurboQuant: Google's KV cache compression
- `3c491c0` — Implement real inference engine: dequantization, forward pass, tokenizer
