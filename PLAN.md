# Implementation Plan

**Project**: PowerInfer_x64 - Pure Rust neuron-sparse LLM inference  
**Target**: Qwen3.5-35B-A3B on 2×GTX 1050 Ti (8GB VRAM) + Jetson Orin Nano  
**Duration**: 48 weeks (12 months)  
**Team**: 1–2 Rust engineers with CUDA/Vulkan experience  

---

## Phase 1: Foundation (Weeks 1–4) — M1

**Goal**: Project scaffolding + CPU inference of tiny model (120M params)

### Tasks

1.1. Repository Setup (Week 1)
- [ ] Create GitHub repo with structure
- [ ] Set up `rust-toolchain.toml` (nightly-2025-06-23)
- [ ] Configure `Cargo.toml` with initial dependencies
- [ ] Add Dockerfile for CUDA 11.8 + rust-gpu toolchain
- [ ] Set up CI: `cargo test`, `cargo fmt`, `cargo clippy`
- [ ] Add pre-commit hooks

1.2. GGUF Loader (Week 2)
- [ ] Implement GGUF v2/v3 parser (use `gguf-rs` as reference)
- [ ] Load metadata: architecture, layers, heads, quantization
- [ ] Mmap weight tensors; lazy loading per tensor
- [ ] Validate magic number, version, checksums
- [ ] Test with TinyLlama-120M GGUF

1.3. CPU Inference Loop (Week 3)
- [ ] Implement tokenizer (BPE; load from GGUF vocab)
- [ ] Build embedding lookup (CPU, no SIMD yet)
- [ ] Implement RMSNorm (f32)
- [ ] Implement RoPE (no SIMD)
- [ ] Implement attention: full (no flash, no KV optimization)
- [ ] Implement FFN: SwiGLU
- [ ] Sampling: greedy + top-p

1.4. Validation (Week 4)
- [ ] Run forward on 100 tokens; compare logits to llama.cpp (within 1e-4)
- [ ] Generate 512 tokens; check coherence manually
- [ ] Benchmark: target >10 tok/s on CPU (Ryzen 9 7950X)
- [ ] Write unit tests for each op

**Deliverable**: `powerinfer-cli` that can run TinyLlama-120M on CPU with acceptable quality.

---

## Phase 2: Quantization & SIMD (Weeks 5–8) — M2

**Goal**: High-performance CPU quantized matmul; support common GGUF quants

### Tasks

2.1. Dequantization Kernels (Week 5)
- [ ] Implement Q4_0: unpack 4-bit→f16
- [ ] Implement Q8_0: unpack 8-bit→f16
- [ ] Implement Q4_K_M: K-quant with K=K, includes superblock scaling
- [ ] Implement Q5_K_M
- [ ] Unit test each dequant against llama.cpp reference bits

2.2. Blocked Matmul (Week 6)
- [ ] Tile sizes: 64×64, 128×128 for L1/L2 cache efficiency
- [ ] Use `std::simd` (f16x8, f32x8) for inner loops
- [ ] Prefetching for large matrices
- [ ] Parallelize outer tiles with `rayon`
- [ ] Benchmark against BLAS (OpenBLAS) for f32

2.3. Quantized Matmul Integration (Week 7)
- [ ] Fuse dequant+matmul: avoid full dequant buffer
- [ ] Support column/row-major weight layout
- [ ] Integrate into FFN forward
- [ ] Test with Qwen2-0.5B Q4_K_M

2.4. CPU Performance Tuning (Week 8)
- [ ] Profile with perf/profiler (cache misses, branch mispredicts)
- [ ] Optimize memory access patterns
- [ ] Target: matmul within 50% of llama.cpp's AVX2 performance
- [ ] Document performance results in `docs/performance.md`

**Deliverable**: CPU backend that runs Qwen3-8B Q4_K_M at >5 tok/s.

---

## Phase 3: Full Architecture (Weeks 9–12) — M3

**Goal**: Complete Qwen3.5 (dense) support on CPU

### Tasks

3.1. RMSNorm + RoPE (Week 9)
- [ ] Vectorized RMSNorm (SIMD)
- [ ] RoPE with partial rotary (apply to head_dim subset)
- [ ] Fuse RMSNorm+RoPE into single pass

3.2. Attention Variants (Week 10)
- [ ] Full attention (causal mask)
- [ ] Gated DeltaNet (autoregressive linear attention)
  - [ ] Implement stateful recurrence
  - [ ] Chunked processing for parallelization
  - [ ] Validate against reference implementation

3.3. MoE Router (Week 11)
- [ ] Top-k routing (k=8–11)
- [ ] Load balancing: auxiliary loss not needed for inference
- [ ] Shared expert (gate)
- [ ] Dispatch tokens to expert buffers

3.4. End-to-End Validation (Week 12)
- [ ] Load Qwen3-8B (dense variant) from GGUF
- [ ] Run perplexity on WikiText-2; must be within 1% of HF baseline
- [ ] Generate 1K tokens; human evaluation for coherence
- [ ] Benchmark: target 2-3 tok/s on CPU (large model)

**Deliverable**: Qwen3-8B runs on CPU with correct outputs.

---

## Phase 4: Profiler & Hot Index (Weeks 13–16) — M4

**Goal**: Build activation profiling pipeline

### Tasks

4.1. Profiler Instrumentation (Week 13)
- [ ] Hook into FFN forward: capture output activations per neuron
- [ ] Compute per-neuron statistics: mean, max, frequency
- [ ] Stream to file (JSONL) to avoid memory blowup
- [ ] Add sampling: random 1% of tokens to reduce overhead

4.2. Profiling Runs (Week 14)
- [ ] Curate dataset: 1000 diverse prompts (from The Pile, RedPajamas)
- [ ] Run profiler on 7B model (CPU)
- [ ] Target throughput: 100 tokens/sec (overhead <20%)
- [ ] Total runtime <4 hours

4.3. Hot Index Builder (Week 15)
- [ ] Parse profiler output; aggregate statistics per neuron
- [ ] For each layer, sort neurons by activation magnitude
- [ ] Select threshold (top p% where p chosen to fit GPU cache)
- [ ] Export as binary format (layer_start, neuron_count, indices)
- [ ] Also expert hotness (for MoE)

4.4. GGUF Extension (Week 16)
- [ ] Define custom GGUF metadata key: `powerinfer.hot_neuron_index`
- [ ] Write serializer to append hot index to GGUF
- [ ] Update loader to read and use hot index
- [ ] Test: hot index loading matches profiler output

**Deliverable**: `powerinfer-profile` tool that produces `model.hot_index.bin`.

---

## Phase 5: Predictor (Weeks 17–20) — M5

**Goal**: Train tiny MLP to predict hot neurons with >95% accuracy

### Tasks

5.1. Training Data Generation (Week 17)
- [ ] From profiler logs: extract (input_activations, hot_labels) pairs
- [ ] Input: top-256 activations from previous layer's FFN output (f16)
- [ ] Output: binary vector over 128 blocks (64 neurons each)
- [ ] Balance dataset: ~1M samples from 1000 prompts
- [ ] Split: 80% train, 10% val, 10% test

5.2. Predictor Architecture (Week 18)
- [ ] Design MLP: 256 → 512 → 128 → 128 (blocks)
- [ ] Use `ndarray` or custom matrix ops
- [ ] Implement forward + backward (SGD)
- [ ] Loss: binary cross-entropy per block (treat as multi-label)
- [ ] Track accuracy, precision, recall

5.3. Training Loop (Week 19)
- [ ] Implement data loader (streaming from disk)
- [ ] SGD with momentum or Adam
- [ ] Learning rate schedule: 0.01 → 0.001
- [ ] Early stopping based on val loss
- [ ] Target: >95% block accuracy on test set

5.4. Predictor Export & Integration (Week 20)
- [ ] Serialize predictor weights as Rust arrays (include_bytes!)
- [ ] Implement predictor inference in runtime (no_std compatible)
- [ ] Benchmark predictor latency (<0.1ms per layer)
- [ ] Validate accuracy in end-to-end setting

**Deliverable**: Trained predictor embedded in `powerinfer-cli` binary.

---

## Phase 6: GPU Offloading — Dense (Weeks 21–24) — M6

**Goal**: Single GPU kernel execution with layer-level offloading

### Tasks

6.1. GPU Backend Setup (Week 21)
- [ ] Add `cust` crate for CUDA driver API
- [ ] Implement `CudaBackend`:
  - [ ] Allocate GPU buffers (pinned host memory for async copies)
  - [ ] Load PTX kernels from embedded bytes
  - [ ] Kernel launch wrapper with argument binding
  - [ ] Stream management (CUDA streams)
  - [ ] Error handling and cleanup
- [ ] Test: simple vector add kernel

6.2. Quantized Dequant Kernels (Week 22)
- [ ] Write CUDA kernels for Q4_0, Q4_K_M, Q5_K_M dequantization
- [ ] Optimize for memory bandwidth (coalesced loads)
- [ ] Use shared memory for dequant tables
- [ ] Verify outputs match CPU reference (epsilon 1e-3)

6.3. Fused Operations Kernel (Week 23)
- [ ] Fuse: dequant → transpose → RMSNorm → RoPE → SwiGLU → matmul
- [ ] Tiling: block matrix multiplication with shared memory
- [ ] Use tensor cores if available (sm_75+); fallback to fused for older
- [ ] Validate: compare to CPU step-by-step (max diff <1e-2)

6.4. Layer Offloading Integration (Week 24)
- [ ] Extend runtime to allocate GPU weights for N layers (`-ngl` equivalent)
- [ ] Copy weights from CPU to GPU at model load (or lazy)
- [ ] For each layer: decide GPU vs CPU based on config
- [ ] Synchronize streams correctly
- [ ] Benchmark: 7B Q4 target 12-15 tok/s on 2×1050 Ti

**Deliverable**: `powerinfer-cli --gpu-layers 28` runs with CUDA backend.

---

## Phase 7: Sparse Neuron Offloading (Weeks 25–28) — M7

**Goal**: Implement predictive hot neuron offloading

### Tasks

7.1. Sparse Matmul Kernel (Week 25)
- [ ] Design kernel: each thread block computes one hot neuron
- [ ] Input: hot indices array, weights matrix (full), activations
- [ ] Coalesced reads from weights (gather)
- [ ] Thread-level parallelism: 256 threads per block
- [ ] Use shared memory for activations broadcast
- [ ] Output write only for hot neurons (scatter)

7.2. Neuron Cache (Week 26)
- [ ] LRU cache tracking which hot neuron blocks are on GPU
- [ ] Cache size = 70% of free GPU memory
- [ ] On miss: async DMA of weight block from CPU pinned memory
- [ ] Prefetch next layer's hot blocks while computing current
- [ ] Cache hit rate target >90%

7.3. Predictor Integration (Week 27)
- [ ] Before each FFN, run predictor on current activations
- [ ] Convert predictor output to hot neuron indices
- [ ] Query neuron cache, schedule missing blocks
- [ ] Launch sparse matmul with hot list
- [ ] CPU fallback: run dequant+matmul for cold neurons

7.4. Performance Tuning (Week 28)
- [ ] Profile kernel occupancy, memory bandwidth utilization
- [ ] Tune hot threshold p% (trade cache hit vs compute waste)
- [ ] Batch multiple layers' hot blocks to reduce kernel launch overhead
- [ ] Measure speedup vs layer-offloading baseline: target 1.5–2×

**Deliverable**: Sparse inference working with >1.5× speedup.

---

## Phase 8: Multi-GPU + MoE (Weeks 29–32) — M8

**Goal**: Run Qwen3.5-35B-A3B on target hardware

### Tasks

8.1. Multi-GPU Coordination (Week 29)
- [ ] Extend `CudaBackend` to manage multiple devices
- [ ] Neuron partitioner: assign hot blocks to GPUs to balance load
- [ ] Bin-packing knapsack algorithm for cache allocation
- [ ] Stream per GPU; cross-GPU synchronization via events

8.2. Large Model Memory Management (Week 30)
- [ ] All model weights stay in CPU RAM (32GB)
- [ ] GPU caches pull weight blocks on-demand (demand paging)
- [ ] Swap out cold blocks using LRU across all GPUs
- [ ] PCIe transfer budgets: limit outstanding copies

8.3. MoE Support (Week 31)
- [ ] MoE router kernel: compute top-8 expert affinities
- [ ] Expert weight caching: entire experts (smaller) kept in GPU cache
- [ ] Dispatch: gather tokens to expert buffers
- [ ] Combine expert outputs

8.4. End-to-End Test (Week 32)
- [ ] Load Qwen3.5-35B-A3B Q4_K_M (~18GB)
- [ ] Configure: total GPU budget 7GB across 2 GPUs
- [ ] Run 100-token generation; measure speed
- [ ] Target: >2 tok/s; VRAM <4GB per GPU; no OOM
- [ ] Compute perplexity on WikiText-2 sample

**Deliverable**: Qwen3.5-35B-A3B runs on 2×GTX 1050 Ti.

---

## Phase 9: Jetson & Vulkan (Weeks 33–36) — M9

**Goal**: Port to ARM64 (Jetson Orin Nano)

### Tasks

9.1. Vulkan Backend (Week 33–34)
- [ ] Implement `VulkanBackend` using `ash`
- [ ] Compile kernels to SPIR-V (rust-gpu)
- [ ] Memory management: Vulkan buffers, mapped memory
- [ ] Descriptor sets for kernel arguments
- [ ] Queue submission with fences

9.2. Kernel Portability (Week 35)
- [ ] Abstract CUDA-specific intrinsics (warp size, sync)
- [ ] Use `cfg(target_os = "cuda")` vs `cfg(target_os = "vulkan")`
- [ ] Test kernels on Vulkan runtime (RenderDoc validation)
- [ ] Performance: within 15% of CUDA on Jetson

9.3. CPU NEON Optimization (Week 36)
- [ ] Use `std::simd` with `target_feature = "neon"` for ARM64
- [ ] Rewrite critical CPU loops with NEON intrinsics if needed
- [ ] Benchmark on Jetson: target 4-6 tok/s for Qwen3-8B

**Deliverable**: Jetson Orin can run Qwen3-8B with Vulkan backend.

---

## Phase 10: Server & Polish (Weeks 37–44) — M10

**Goal**: Production-ready API and user experience

### Tasks

10.1. HTTP Server (Week 37–38)
- [ ] Implement `/v1/completions` (POST)
- [ ] Implement `/v1/chat/completions` with streaming SSE
- [ ] Chat templates: support Qwen chat format (Jinja2-style)
- [ ] Model management: hot-reload on SIGHUP
- [ ] Rate limiting, request queueing (max_concurrent)
- [ ] Prometheus metrics endpoint

10.2. Configuration & CLI (Week 39)
- [ ] Config file (TOML) for server: model path, GPU layers, cache size, concurrency
- [ ] CLI flags for all modes
- [ ] `powerinfer-serve` wrapper
- [ ] Logging: structured JSON with tracing

10.3. Benchmark Suite (Week 40–41)
- [ ] Prompt set: 10 diverse prompts (coding, reasoning, creative)
- [ ] Metrics: tokens/sec, memory usage, VRAM, latency p50/p99
- [ ] Compare against llama.cpp (same model, best ngl)
- [ ] Document results in `docs/performance.md`

10.4. Testing & CI (Weeks 42–43)
- [ ] GitHub Actions: test on Ubuntu (CPU, CUDA)
- [ ] Add integration tests with small model
- [ ] Fuzz testing: random inputs to kernels
- [ ] Memory leak checks (Valgrind, sanitizers)

10.5. Documentation (Week 44)
- [ ] README: quickstart, features, benchmarks
- [ ] Build guide: CUDA, Vulkan, Docker
- [ ] Tuning guide: hot threshold, cache size, GPU layers
- [ ] Troubleshooting: OOM, slow performance, accuracy issues
- [ ] API reference (docs.rs)

**Deliverable**: v1.0 beta release; public benchmark results.

---

## Phase 11: Release (Weeks 45–48) — M11

**Goal**: Stabilize and ship v1.0

### Tasks

11.1. Bug Fixes (Weeks 45–46)
- [ ] Address all critical GitHub issues (crash, OOM, wrong outputs)
- [ ] Memory leak elimination
- [ ] Accuracy regression testing (perplexity)
- [ ] Edge cases: long context (32K), very small prompts

11.2. Optimization Pass (Week 47)
- [ ] Profile realistic workloads (interactive chat)
- [ ] Reduce kernel launch overhead (batching, fusion)
- [ ] Tune default hyperparameters (threshold, cache size)
- [ ] Add auto-tuning: benchmark cache sizes on startup

11.3. Release Preparation (Week 48)
- [ ] Update version to 1.0.0
- [ ] Build and test release binaries (Linux x86_64, aarch64)
- [ ] Publish Docker images to GitHub Container Registry
- [ ] Create release notes with benchmarks
- [ ] Tag v1.0.0; announce on forums (r/LocalLLaMA, GitHub Discussions)

**Deliverable**: v1.0.0 release on GitHub.

---

## Milestone Summary

| Milestone | Deliverable | Week |
|-----------|-------------|------|
| M1 | CPU 120M inference | 4 |
| M2 | CPU Q4_K_M matmul | 8 |
| M3 | Qwen3-8B CPU | 12 |
| M4 | Profiler + hot index | 16 |
| M5 | Predictor >95% acc | 20 |
| M6 | Single GPU layer-offload | 24 |
| M7 | Sparse neuron offload | 28 |
| M8 | Multi-GPU + MoE + 35B | 32 |
| M9 | Jetson Vulkan backend | 36 |
| M10 | Server + benchmarks | 44 |
| M11 | v1.0 release | 48 |

---

## Risk Register

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| rust-gpu toolchain unstable | High | High | Use pinned Docker image; contribute fixes upstream; fallback to C++ kernels if needed |
| Predictor accuracy <90% | High | Medium | Increase predictor size; use more profiles; add adaptive fallback thresholds |
| PCIe bandwidth saturation | Medium | High | Batch transfers; use pinned memory; overlap compute/comm |
| Qwen3.5 conversion complexity | High | Medium | Use community conversion scripts; validate against HF reference |
| Time estimate optimistic | High | High | Buffer 20% time per phase; prioritize MVP features |
| Pure Rust CUDA performance gap | Medium | Medium | Benchmark early (M6); fall back to llama.cpp kernels via FFI if gap >50% |

---

## Success Metrics

- **Functional**: Qwen3.5-35B-A3B generates coherent text on 2×1050 Ti
- **Performance**: ≥2 tok/s (vs llama.cpp 1 tok/s on same hardware)
- **Memory**: <4GB VRAM per GPU; no swapping
- **Quality**: Perplexity within 2% of dense baseline
- **Usability**: `docker run` gets working server in <10 minutes

---

## Conclusion

This plan is ambitious but achievable with disciplined execution. The key is hitting M6 (GPU offload) early to validate the rust-gpu approach, then iterating on sparse optimizations.
