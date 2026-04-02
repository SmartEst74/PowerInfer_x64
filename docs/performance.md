# Performance Guide

This document describes the current benchmark baseline, how to report performance honestly, and what is still blocking the repo's target throughput.

## Targets

These remain the repo goals. They are not the current measured throughput.

| Model | Hardware | VRAM | Target tok/s |
|-------|----------|------|--------------|
| Qwen3.5-35B-A3B Q4 | 2× GTX 1050 Ti | 7.5GB | 2.5–4 |
| Qwen3-8B Q4 | 2× GTX 1050 Ti | 5GB | 12–16 |
| Llama2-7B Q4 | 2× GTX 1050 Ti | 4.5GB | 15–20 |
| Qwen3-8B Q4 | Jetson Orin Nano | 6GB shared | 4–6 |

## Current Verified Baseline

Benchmark command:

```bash
cargo run --release --bin real_test -- /path/to/model.gguf
```

Latest verified result:

| Metric | Result |
|--------|--------|
| Model | Qwen3.5-35B-A3B Q8_0 |
| Backend | CPU |
| GGUF parse | 1.85s |
| Inference context build | 3.57s |
| Prefill | 3.39s |
| Decode average | 1.44-1.46 tok/s |
| Best observed token | 1.71 tok/s |
| Output first token | `Paris` |

Hardware used for this baseline:
- Pentium G4400, 2 cores, SSE4.2, no AVX2
- 32 GB DDR4
- 2× GTX 1050 Ti present, but validated generation still runs on the CPU backend
- PCIe Gen1 x8 and x4 links reported by the execution-plan scan

## Rules For Reporting Performance

- Use `--release` for any throughput or latency claim.
- Always report the exact model file, hardware, and backend used.
- Distinguish between planned GPU placement and actual runtime dispatch.
- Do not present server request throughput as inference throughput by default; use explicit end-to-end server measurements and report the sampling settings (`temperature`, `top_p`) used for that run.
- Do not claim GPU speedup until issues #128 and #129 are resolved and measured on real hardware.

## Why The Repo Is Not At Target Yet

| Bottleneck | Current impact |
|------------|----------------|
| End-to-end CUDA dispatch not wired | CPU remains the validated execution backend |
| Sparse hot-neuron scheduling not implemented | Core PowerInfer speedup path is still missing |
| Old 2-core host CPU | Limits parallelism |
| SSE4.2 but no AVX2/FMA | CPU decode is materially slower than on newer x86 hosts |
| Slow PCIe links on both 1050 Ti cards | Makes naive per-token transfer strategies especially expensive |
| LM head remains expensive | Release traces show roughly 67-117ms in the LM head alone |

Release-mode timing traces from the latest validation run show per-layer timings around 13-21ms for the early layers, 11-18ms for the final layer, and roughly 67-117ms for the LM head. That aligns with the observed ~693ms decode average.

## Optimizations Already Applied

| Optimization | Effect |
|--------------|--------|
| SSE4.1 Q8_0 matvec | ~2x vs scalar |
| Parallel expert FFN | ~2x expert throughput |
| Parallel large-output matvec | ~1.7x on QKV and LM head |
| Batch mmap prefetch | Fewer page faults |
| Zero-copy expert access | Less allocation overhead |
| Hardware-adaptive execution plan | Better CPU/GPU placement awareness |
| TurboQuant KV integration | Present in the runtime path |

## Reproducing The Baseline

### Full benchmark

```bash
cargo run --release --bin real_test -- /path/to/model.gguf
```

### Minimal generation smoke test

```bash
cargo run --release --bin powerinfer-cli -- generate \
    --model /path/to/model.gguf \
    --prompt "The capital of France is" \
    -n 1
```

### Metadata verification

```bash
cargo run --release --bin gguf_dump -- /path/to/model.gguf
```

## What To Improve Next

1. Wire real CUDA execution into the runtime and prove end-to-end token generation on GPU.
2. Implement sparse hot-neuron execution instead of stopping at execution planning.
3. Add reference comparisons against llama.cpp for the validated Qwen3.5 path.
4. Add benchmark CI so future performance changes are visible immediately.
