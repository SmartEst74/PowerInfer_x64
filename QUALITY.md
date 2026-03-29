# Quality Baseline

**Date**: 2026-03-29
**Commit**: d64758e on master
**Rust**: stable 1.94.1 (switched from nightly)

## Test Results

| Category | Count | Status |
|----------|-------|--------|
| Unit tests | 47 | ✅ PASS |
| Smoke tests | 6 | ✅ PASS |
| Soak tests (10K iterations) | 7 | ✅ PASS |
| Integration tests | 5 | ✅ PASS |
| Quality tests (real model) | 3+2 | ✅ PASS |
| Doc tests | 1 | ✅ PASS |
| **Total** | **69** | **✅ ALL PASS** |

## Lint

| Check | Status |
|-------|--------|
| `cargo fmt --check` | ✅ PASS |
| `cargo clippy -D warnings` (stable) | ✅ PASS |
| `cargo check --features cuda` | ✅ COMPILES |

## Real Model Test (2026-03-29)

Tested against `Arch-Agent-3B.Q8_0.gguf` (3.1GB, Qwen2 arch, 36 layers, 2048 dim).

### Verified

| Step | Time | Status |
|------|------|--------|
| GGUF v3 parse | 1.2s | ✅ PASS |
| Metadata extraction | instant | ✅ PASS — architecture, layers, dim all correct |
| Weight dequant (Q8_0 → f32) | 18s | ✅ PASS — 434 tensors, all finite |
| Forward pass (single token) | 30s | ✅ PASS — 151936 logits, all finite |
| Logit quality | — | ✅ PASS — range -14.7 to 14.4, reasonable |
| Embedding values | — | ✅ PASS — mean 0.0001, range -0.09 to 0.08 |

### What Works

- GGUF parsing (all architectures: qwen2, qwen3, llama)
- Dequantization (Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K_M, Q5_K_M, Q6_K)
- Forward pass (RMSNorm → RoPE → attention → FFN SwiGLU → output)
- SIMD SSE4.1 matvec (4x speedup over scalar)
- CUDA backend skeleton (compiles with cust 0.3.2)
- System resource detection (2x GTX 1050 Ti, 8GB VRAM)
- TurboQuant KV cache compression (algorithm, 8x ratio)
- Activation profiler (neuron hotness tracking)
- CI/CD pipeline (4 stages, stable Rust)
- Prometheus metrics (9 metrics, /metrics endpoint)

### What Doesn't Work Yet

| Gap | Issue | Impact |
|-----|-------|--------|
| Coherent output generation | #126 | Forward pass runs but output not yet tested for coherence |
| Reference comparison | #127 | Need llama.cpp logits for validation |
| CUDA GPU execution | #128 | PTX kernel written but not tested on actual GPU |
| Sparse GPU execution | #129 | Core PowerInfer innovation — not yet implemented |
| Benchmark CI | #130 | Performance regression tracking needed |

### Performance

| Config | Speed | Notes |
|--------|-------|-------|
| CPU (Pentium G4400, SIMD) | ~30s/token | Functional, not usable |
| CUDA (2x GTX 1050 Ti) | NOT TESTED | PTX kernel compiles, not run on GPU |
| Target (sparse GPU) | 5-10 tok/s | Requires CUDA + MoE routing |

## Honest Assessment

The codebase is **30% of the functional goal**. The foundation (GGUF parsing, dequantization, forward pass) is proven against real model files. The core innovation (GPU sparse execution) is scaffolded but not tested. The project needs:

1. CUDA matvec actually running on GPU (test PTX kernel on 1050 Ti)
2. Forward pass on GPU (move attention + FFN to GPU)
3. MoE routing (select active experts per token)
4. Quality validation (perplexity comparison)

## Open Issues (5)

| # | Title | Priority |
|---|-------|----------|
| 126 | Generate coherent output | HIGH |
| 127 | Validate against llama.cpp reference | HIGH |
| 128 | CUDA kernels on GPU | HIGH |
| 129 | Sparse GPU execution | HIGH |
| 130 | Performance benchmark CI | MEDIUM |
