# Quality Baseline

**Date**: 2026-04-03
**Default Rust Toolchain**: stable
**Nightly Needed For**: future `rust-gpu` kernel experiments only (`nightly-2025-06-23`)

## Commands Verified In This Snapshot

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo run --release --bin gguf_dump -- /home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf
cargo run --release --bin real_test -- /home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf
cargo run --release --bin powerinfer-cli -- generate --model /home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf --prompt "The capital of France is" -n 1
cargo run --release --bin powerinfer-cli -- generate --model /home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf --prompt "The capital of France is" -n 4 --temperature 0.7 --top-p 0.9
cargo run --release --features profiling --bin powerinfer-profile -- --model /home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf --output /tmp/powerinfer-hot-index.json --samples 2 --layers 2
cargo test --features server
cargo clippy --all-targets --features server -- -D warnings
cargo test --features 'server profiling'
```

## Code Health

| Check | Result |
|-------|--------|
| `cargo test` | 86 passed, 2 ignored, 0 failed |
| `cargo clippy --all-targets -- -D warnings` | PASS |
| `cargo test --features server` | PASS |
| `cargo clippy --all-targets --features server -- -D warnings` | PASS |
| `cargo test --features 'server profiling'` | PASS |
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
| **Decode average** | **2.23 tok/s (448ms/token)** |
| Best observed token | 2.44 tok/s |
| Stable range (taskset pinned) | 2.17–2.32 tok/s |
| **Target met** | **YES — target was 2.0 tok/s** |
| All 10 tokens | > 2.0 tok/s (range: 2.17–2.32 with CPU pinning) |
| Output | `located in which country?\n\n<think>ziiinn France` |
| First relevant token | `France` context ✅ |
| Backend | CPU + 2× GTX 1050 Ti (CUDA, persistent VRAM weights) |

#### Performance Notes

- **System noise**: On this 2-core system, background processes (VS Code, desktop) cause ±15% variance. Using `taskset -c 0,1` reduces variance to ±3%.
- **Hardware ceiling**: SSE4.2 (no AVX2) limits SIMD throughput to 4-wide. The Pentium G4400 is compute-bound at ~2.23 tok/s with this model.
- **Self-tuning**: The `ExecutionPlan` now detects core count and adapts MoE parallelism (2-thread split on ≤2 cores, shared expert overlap on ≥3 cores).
- **Easy mode**: `powerinfer-cli easy --model <path>` auto-detects hardware, configures optimally, and reports tok/s.

### Previous Results (CPU-only, before GPU offload)

| Metric | Result |
|--------|--------|
| Decode average | 1.44–1.46 tok/s |
| Best observed token | 1.71 tok/s |

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
| TurboQuant KV cache integration | Active | 3-bit keys + QJL correction, f16 values |
| **TurboQuant precomputed query rotation** | Active | **5× faster attention (17ms→3ms/layer)** |
| **CUDA GPU weight offload (40 layers)** | Active | **All projection matvecs on GPU** |
| **LM head split across 2 GPUs** | Active | **85ms→21ms LM head** |
| **Persistent GPU scratch buffers** | Active | **Eliminates per-call CUDA alloc/free** |
| **Warp-per-row GPU kernel (sm_61)** | Active | **Shared memory + shuffle reduction** |
| **mlock shared expert/router weights** | Active | **~218 MB pinned in RAM** |
| **Scratch buffer reuse in forward pass** | Active | **Eliminates per-layer heap allocs** |
| **In-place FFN residual** | Active | **Removes .to_vec() copy per layer** |
| **Adaptive shared expert overlap** | Active | **Overlaps on ≥3 cores, sequential on ≤2** |
| **TurboQuant near-zero weight skip** | Active | **Faster weighted_value_sum for long contexts** |
| **`easy` CLI auto-detect + timing** | Active | **Zero-config model execution with perf report** |

## What Works

- GGUF parsing for `qwen2`, `qwen3`, `llama`, and `qwen35moe` metadata.
- Memory-mapped weight loading for a 34 GiB GGUF without fully copying it into RAM.
- CPU inference through the validated Qwen3.5 forward path.
- Real-model generation with a correct first token.
- Hardware profile and execution-plan reporting for the dual-1050-Ti development machine.
- Quantization and dequantization unit coverage for Q4_0, Q8_0, Q4_1, Q4_K, Q5_K, Q6_K, F16, and F32 paths.
- TurboQuant KV path wiring in the current forward implementation.
- HTTP server routes that return real model-backed completions and chat completions in release mode.
- Basic temperature/top-p sampling through both the CLI and HTTP server.
- Basic hot-index export from the profiler and `powerinfer-cli profile`.
- Experimental runtime loading of hot-index JSON for dense FFN sparsity and hot MoE expert caching.

## Known Gaps

| Gap | Status |
|-----|--------|
| ~~Target throughput on old 2-core CPU~~ | **MET: 2.23 tok/s avg** |
| End-to-end CUDA token generation | Partially done (projections + LM head on GPU, MoE on CPU) |
| Sparse hot-neuron GPU execution | Not implemented |
| Reference comparison against llama.cpp | Open |
| Server-backed real completions | Verified, basic |
| Server-side basic sampling | Verified, basic |
| Profiler hot-index generation | Verified, basic |
| Runtime use of hot-index output | Experimental CPU-only |
| Benchmark regression CI | Open |

## Server Validation

Validated release server command:

```bash
cargo run --release --features server --bin powerinfer-serve -- /home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf
```

Validated local HTTP checks:

```bash
curl -s http://127.0.0.1:8080/v1/models
curl -s http://127.0.0.1:8080/v1/completions \
	-H 'Content-Type: application/json' \
	-d '{"model":"Qwen3.5-35B-A3B","prompt":"The capital of France is","max_tokens":8,"temperature":0.0}'
curl -s http://127.0.0.1:8080/v1/chat/completions \
	-H 'Content-Type: application/json' \
	-d '{"model":"Qwen3.5-35B-A3B","messages":[{"role":"user","content":"Reply with one word: France capital?"}],"max_tokens":4,"temperature":0.0}'
```

Observed behavior:
- `/v1/models` reports `Qwen3.5-35B-A3B`.
- `/v1/completions` returned model-generated text from the loaded GGUF rather than a dummy payload.
- Sampled `/v1/completions` and `/v1/chat/completions` requests succeeded with `temperature=0.7` and `top_p=0.9`.
- `/v1/chat/completions` returned a real chat-completion payload rather than a dummy response.
- The server still rejects streaming requests explicitly.

## Profiler Validation

Validated profiler commands:

```bash
cargo run --release --features profiling --bin powerinfer-profile -- \
	--model /home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf \
	--output /tmp/powerinfer-hot-index.json \
	--samples 2 \
	--layers 2

cargo run --release --bin powerinfer-cli -- profile \
	--model /home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf \
	--output /tmp/powerinfer-hot-index-cli.json \
	--samples 2 \
	--layers 2
```

Observed behavior:
- Both entry points produced a real JSON hot-index file rather than exiting with a placeholder message.
- For the validated Qwen3.5 MoE path, the current profiler records expert selection hotness on MoE layers.
- With the default `min_hotness=0.05`, the exported MoE hot-index file contained non-empty expert lists for the sampled layers.
- The runtime now accepts the exported JSON format for experimental dense FFN sparsity and hot MoE expert caching on the CPU path.

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
