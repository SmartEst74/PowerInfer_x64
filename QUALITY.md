# Quality Baseline

**Date**: 2026-04-02
**Default Rust Toolchain**: stable
**Nightly Needed For**: future `rust-gpu` kernel experiments only (`nightly-2025-06-23`)

## Commands Verified In This Snapshot

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo run --release --bin gguf_dump -- /home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf
cargo run --release --bin real_test -- /home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf
cargo run --release --bin powerinfer-cli -- generate --model /home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf --prompt "The capital of France is" -n 1
```

## Code Health

| Check | Result |
|-------|--------|
| `cargo test` | 81 passed, 2 ignored, 0 failed |
| `cargo clippy --all-targets -- -D warnings` | PASS |
| Ignored tests | 2 path-dependent quality checks for alternate local GGUFs |

## Current Real-Model Benchmark

**Model**: `Qwen3.5-35B-A3B-Q8_0.gguf`
**Architecture**: `qwen35moe`
**Model facts**: 34.37 GiB, 733 tensors, 40 layers, 256 experts, 8 active experts
**Hardware**: Pentium G4400, 32 GB DDR4, SSD, 2× GTX 1050 Ti present
**Backend used for validated run**: CPU

### Release-Mode Results

| Metric | Result |
|--------|--------|
| GGUF parse | 1.85s |
| Inference context build | 3.57s |
| Prefill | 3.39s |
| Decode average | 1.44-1.46 tok/s across repeated release runs |
| Best observed token | 1.71 tok/s |
| Output | `Paris.ĊChooseĠtheĠcorrectĠanswerĠbelow:Ġmassive` |
| First token check | `Paris` ✅ |

### Debug-Mode Warning

The same `real_test` path in the dev profile was roughly `0.02 tok/s`. Debug builds are valid for debugging, not for performance claims.

## Output Verification

Prompt:

```text
The capital of France is
```

Observed output:

```text
Paris.ĊChooseĠtheĠcorrectĠanswerĠbelow:Ġmassive
```
Quality interpretation:
- The first generated token is correct.
- The continuation is not yet fully clean or chat-ready.
- This is enough to prove the forward path is producing plausible output, but not enough to claim production generation quality.

## Critical Correctness Fixes That Unblocked This Result

| Fix | Impact |
|-----|--------|
| Added missing `1/sqrt(k_hd)` query scaling after L2 normalization | Fixed attention magnitude |
| Fixed tiled V-head ordering and untiled `A_log` / `dt_bias` mapping | Fixed corrupted Gated Delta Rule behavior |
| Stopped double-exponentiating `ssm_a` | Fixed state decay |
| Switched MoE routing from sigmoid to softmax -> top-k -> renormalize | Fixed expert weighting |

## Optimizations Applied

| Optimization | Status | Observed effect |
|--------------|--------|-----------------|
| SSE4.1 Q8_0 matvec (`cvtepi8 -> cvtepi32 -> f32`) | Active | ~2x vs scalar |
| Parallel expert FFN (2 threads, 4 experts each) | Active | ~2x expert throughput |
| Parallel matvec for large outputs (`n_out >= 4096`) | Active | ~1.7x on QKV and LM head |
| Batch mmap prefetch for active experts | Active | Fewer page faults |
| Zero-copy expert weight access (`Arc<Mmap>`) | Active | Avoids per-expert allocations |
| Hardware-adaptive execution planning | Active | Computes a CPU/GPU layer split and memory plan |
| TurboQuant KV cache integration | Active | Enabled in the current execution plan |

## What Works

- GGUF parsing for `qwen2`, `qwen3`, `llama`, and `qwen35moe` metadata.
- Memory-mapped weight loading for a 34 GiB GGUF without fully copying it into RAM.
- CPU inference through the validated Qwen3.5 forward path.
- Real-model generation with a correct first token.
- Hardware profile and execution-plan reporting for the dual-1050-Ti development machine.
- Quantization and dequantization unit coverage for Q4_0, Q8_0, Q4_1, Q4_K, Q5_K, Q6_K, F16, and F32 paths.
- TurboQuant KV path wiring in the current forward implementation.

## Known Gaps

| Gap | Status |
|-----|--------|
| Target throughput on old 2-core CPU | Not met |
| End-to-end CUDA token generation | Not wired |
| Sparse hot-neuron GPU execution | Not implemented |
| Reference comparison against llama.cpp | Open |
| Server-backed real completions | Not implemented |
| Profiler hot-index generation | Not implemented |
| Benchmark regression CI | Open |

## Open Issues

| Issue | Focus |
|------|-------|
| #126 | Improve generation quality |
| #127 | Validate against llama.cpp reference |
| #128 | Wire CUDA kernels into real execution |
| #129 | Implement sparse GPU execution |
| #130 | Add performance benchmark CI |

## Historical Milestone

Previous real-model validation on 2026-03-29 used `Arch-Agent-3B.Q8_0.gguf` and proved the code could parse GGUF, load weights, and run a forward pass. That older local model is not present in the current environment, so it remains historical context rather than the current benchmark baseline.
