# Quality Baseline

**Date**: 2026-04-01 (updated)
**Rust**: nightly-2025-06-23

## Test Results

| Category | Count | Status |
|----------|-------|--------|
| Unit tests | 3 | ✅ PASS |
| Smoke tests | 6 | ✅ PASS |
| Soak tests | 7 | ✅ PASS |
| Doc tests | 1 | ✅ PASS |
| **Total** | **17** | **✅ ALL PASS** |

## Lint

| Check | Status |
|-------|--------|
| `cargo clippy --all-targets -- -D warnings` | ✅ PASS |
| No GPU backend | ⚠️ INFO — CPU-only mode |

## Real Model Test: Qwen3.5-35B-A3B-Q8_0 (2026-04-01, latest)

Tested against `Qwen3.5-35B-A3B-Q8_0.gguf` (34.37 GiB, 733 tensors, 40 layers).
Hardware: Pentium G4400 (2 cores, 3.3 GHz, SSE4.2, no AVX), 32GB DDR4, SSD, no GPU used.

### Performance

| Metric | Value |
|--------|-------|
| GGUF parse | 1.8s |
| Weight loading (mmap) | ~0s (zero-copy) |
| Prefill (7 tokens) | 3.2–3.8s |
| **Decode (avg)** | **1.37–1.54 tok/s** |
| Decode (best token) | 1.62 tok/s |
| Output | **"Paris"** ✅ CORRECT |

### Output Verification

Prompt: `"The capital of France is"`
Output: `"Paris.ĊChooseĠtheĠcorrectĠanswerĠbelow:Ġmassive"`
First token: **"Paris"** — factually correct answer.

### Bugs Fixed (this session — 4 critical fixes)

| Bug | Fix | Impact |
|-----|-----|--------|
| Missing Q scale factor | Added `1/sqrt(k_hd)` scale after L2-norm | CRITICAL — wrong attention magnitude |
| Tiled V head ordering wrong | Fixed `kh = vh % n_k_h` + un-tiled A_log/dt_bias mapping | CRITICAL — garbled output |
| ssm_a double-exponential | Use GGUF value directly (already = -exp(A_log)), don't call .exp() | CRITICAL — wrong GDR decay |
| MoE routing sigmoid instead of softmax | softmax over all 256 experts → topk → renormalize | CRITICAL — wrong expert weighting |

### Optimizations Applied

| Optimization | Speedup |
|-------------|---------|
| SSE4.1 Q8_0 matvec (cvtepi8→cvtepi32→f32) | 2x vs scalar |
| Parallel expert FFN (2 threads, 4 experts each) | ~2x expert throughput |
| Parallel matvec for large matrices (n_out ≥ 4096) | ~1.7x for QKV/LM head |
| Batch mmap prefetch (all 8 experts before compute) | Reduced page faults |
| Zero-copy expert weight access (Arc\<Mmap\>) | No allocation per expert |

### Hardware Speed Limit Analysis

| Component | Per Token |
|-----------|-----------|
| 40 layers × ~15ms each | ~600ms |
| LM head (248320 outputs, parallel) | ~75ms |
| **Total compute** | **~675ms → 1.48 tok/s** |
| **Theoretical max (2 cores, perfect)** | **~2 tok/s** |

**Why 10 tok/s is not achievable on this hardware:**
- Active parameters per token: ~3.2 GB (8 experts × 40 layers + attention + LM head)
- CPU throughput: ~4.4 GFLOPS effective (2 cores × SSE4.1, Q8_0 decode overhead)
- Memory bandwidth: ~20 GB/s (DDR4-2133 dual channel)
- Minimum compute time: 3B ops / 4.4 GFLOPS = ~680ms → 1.47 tok/s
- 10 tok/s requires GPU offloading (GTX 1050 Ti: ~1075 GFLOPS FP32)

### Architecture Notes

- **qwen35moe**: Hybrid architecture with Gated Delta Rule (linear attention) + full-attention + MoE FFN
- **Layer types**: 30 GDR + 10 full-attention (every 4th layer starting at 3), all with 256-expert MoE
- **Full attention**: Q weight packs [query, gate] interleaved per head; gate = sigmoid applied to attention output
- **GDR mechanism**: Delta rule with [32 heads × 128 × 128] state matrix, conv1d→SiLU, L2-normalized Q/K, Q scaled by 1/sqrt(k_hd), RMSNormGated output
- **MoE routing**: softmax over 256 experts → top-8 → renormalize by sum (NOT sigmoid)
- **Memory**: mmap-based zero-copy (tensor data stays on disk, OS pages on demand)
- **Expert tensors**: 3×256 packed expert matrices per layer, extracted per-expert via zero-copy mmap slices

### What Works

- GGUF parsing (all architectures: qwen2, qwen3, llama, qwen35moe)
- Memory-mapped weight loading — 34 GiB model runs in 5 GB RAM
- **Correct output**: "Paris" for "The capital of France is" ✅
- Full attention with Q/gate de-interleave and sigmoid output gating
- Gated Delta Rule with state matrix [32,128,128], tiled V head ordering
- MoE routing with softmax + renormalize (correct Qwen3.5 routing)
- Shared expert with learned dot-product gate
- Parallel expert FFN (2 threads) + parallel matvec (≥4096 outputs)
- SSE4.1 optimized Q8_0 matvec kernel
- Residual connections, RMSNorm (weight+1 baked into GGUF), RoPE

### Known Gaps

| Gap | Issue | Priority |
|-----|-------|----------|
| Speed | 1.4 tok/s on CPU (need GPU offload for 10+ tok/s) | HIGH |
| GPU offloading | ExecutionPlan computed but CUDA dispatch not wired | HIGH |
| Temperature/top-p | Only greedy (argmax) implemented | MED |
| Chat template | No Qwen3.5 prompt formatting | MED |

### Previous Real Model Test: Arch-Agent-3B.Q8_0 (2026-03-29)

| Step | Time | Status |
|------|------|--------|
| GGUF v3 parse | 1.2s | ✅ PASS |
| Forward pass (single token) | 30s | ✅ PASS — 151936 logits, all finite |

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
- TurboQuant KV cache compression (V2 with QJL; V3 MSE-only is superior, see docs/architecture.md)
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
