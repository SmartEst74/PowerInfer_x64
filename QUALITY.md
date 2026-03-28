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

## Real Model Test (2026-03-28)

Tested against `Arch-Agent-3B.Q8_0.gguf` (3.1GB) on CPU (Ryzen 9 7950X).

### Results

| Step | Time | Status |
|------|------|--------|
| GGUF file load | 1.36s | ✅ PASS |
| Metadata parse | instant | ✅ PASS — qwen2 arch, 36 layers, 2048 dim, 16 heads, 2 KV heads |
| Tensor listing | instant | ✅ PASS — 438 tensors, kind=8 (Q8_0) |
| Weight load (Q8_0 → f32) | 96s | ✅ PASS — dequantizes all 3.1GB |
| Forward pass | very slow | ⚠️ PARTIAL — runs but too slow without SIMD |

### What Works

- GGUF v3 parsing with `qwen2` architecture prefix (not hardcoded to `llama.`)
- Q8_0 dequantization of real model weights (3.1GB → f32)
- Model config extraction from real GGUF metadata
- Tensor name-based weight lookup
- Full forward pass executes (RMSNorm → Q/K/V → attention → FFN → output)

### What's Slow

- Pure f32 matvec without SIMD: each matvec is O(n²) with no vectorization
- A 2048×11008 FFN matvec = 22.5M multiply-adds, ~10ms per call on CPU
- 36 layers × 4 matvecs per layer × 2 passes (prompt + decode) = very slow
- Needs: SIMD (std::simd), tiled matmul, fused dequant+matmul (Issue #40-45)

### What's Missing for Qwen3-4B

- Q5_K dequantization not implemented (kind=12) — Qwen3-4B uses Q4_K_M which requires Q5_K support
- Q4_K_M dequantization not implemented (superblock scaling)
- attn_k_norm and attn_q_norm tensors present in Qwen3 but not used in forward pass
- No Qwen3-specific full_attention_interval handling

## Honest Assessment

The pipeline **works end-to-end** for Q8_0 models:
- Parse GGUF ✅
- Read metadata ✅
- Load weights ✅
- Run forward pass ✅ (functional, not performant)

For Q4_K_M models (Qwen3-4B, most common quant):
- Parse GGUF ✅
- Read metadata ✅
- Load weights ❌ (Q5_K dequant not implemented)

Next priority: implement Q4_K_M and Q5_K dequantization to support the most common model format.

## What Is NOT Yet Tested

| Requirement | Issue | Status |
|-------------|-------|--------|
| Forward pass vs llama.cpp reference (1e-4) | #31 | ❌ NOT DONE |
| Generate 512 tokens coherence check | #32 | ❌ NOT DONE |
| Qwen3-8B perplexity validation | #60 | ❌ NOT DONE |
| Benchmark: >10 tok/s on CPU | #33 | ❌ NOT DONE |
| Q4_K_M dequantization | #37 | ❌ NOT DONE |
| Q5_K_M dequantization | #38 | ❌ NOT DONE |

## CI Pipeline

- Lint: `cargo fmt` + `cargo clippy -D warnings` (both default and server features)
- Test: matrix strategy (no features + --features server)
- Security: `cargo audit`
- Benchmark: runs on push, stores artifacts

## Commit History (this session)

- Real model test with Arch-Agent-3B Q8_0 — GGUF loads, weights load, forward pass runs
- Fix GGUF config parser to support architecture-specific prefixes (qwen2, qwen3, etc.)
- Add smoke/soak/integration test suites (18 new tests)
- Close 19 completed GitHub issues
- CI matrix strategy for feature testing
- TurboQuant KV cache compression (7 tests)
- Activation profiler (6 tests)
- Quality benchmark (4 tests)
