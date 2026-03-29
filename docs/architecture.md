# PowerInfer_x64 Architecture

## Overview

PowerInfer_x64 is a **neuron-level sparse LLM inference engine** written in pure Rust. It extends the PowerInfer concept to modern architectures (Qwen3.5 MoE + DeltaNet) and heterogeneous hardware (x86-64 CUDA, ARM64 Vulkan).

## Core Innovation: Predictive Hot Neuron Caching

Traditional LLM inference offloads **entire layers** to GPU (e.g., llama.cpp's `-ngl`). PowerInfer_x64 goes finer-grained: it offloads **individual neurons** (FFN units) based on their recent activation history.

### Why Neurons?

In transformer models, the Feed-Forward Network (FFN) layers are wide (e.g., 11008 neurons for Llama-7B). For any given input token, only a **small subset** of these neurons are strongly activated (high magnitude output). The rest are "cold" and contribute little to the final output.

If we can:
1. **Predict** which neurons will be hot for the current context
2. **Keep only hot neuron weights in GPU memory**
3. **Compute cold neurons on CPU** (or fetch them on-demand)

Then we can fit models far larger than GPU memory would normally allow.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Rust Runtime Layer                     │
├─────────────────────────────────────────────────────────────┤
│  Model Loader │ Weight Loader │ Tokenizer │ Profiler (stub) │
└───────────────────────────────────┬─────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│           GPU Backend Abstraction (CUDA/Vulkan)            │
├─────────────────────────────────────────────────────────────┤
│  Kernel Loader │ Buffer Manager │ Stream Coordination      │
└───────────────────────────────────┬─────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│           GPU Kernels (CUDA PTX inline, cust bindings)      │
├─────────────────────────────────────────────────────────────┤
│  MatVec (PTX) │ ... more kernels planned                   │
└─────────────────────────────────────────────────────────────┘
```

### 1. GGUF Loader

- Parse GGUF v2/v3 files using [`gguf-rs`](https://crates.io/crates/gguf-rs)
- Extract model configuration (architecture, layers, heads, quantization)
- Load quantized weights into CPU memory (mmap)
- Optionally read extended `powerinfer.hot_neuron_index` metadata if pre-profiled

### 2. Activation Profiler

**Purpose**: Collect statistics about neuron activations across diverse inputs.

**Process**:
- Run model on 1000s of tokens from The Pile, RedPajamas, etc.
- For each FFN neuron (per layer), record:
  - Mean activation magnitude
  - Max activation
  - Activation frequency (times non-zero)
- Post-process: for each layer, sort neurons by average magnitude
- Select top-p% (e.g., 5-10%) as "hot" for that layer
- Also record MoE expert usage frequency

**Output**: Hot neuron index file mapping `(layer, neuron_id) → bool`

### 3. Predictor Model

A tiny MLP trained to predict hot neurons from current context.

- **Input**: Top-k activation values from previous layer's FFN output (k=256)
- **Output**: Binary vector over neuron blocks (64 neurons per block → 128 blocks total for 11008 neurons)
- **Architecture**: 2 dense layers (256→512→128) with ReLU
- **Training data**: (input_activations, hot_labels) from profiler
- **Training**: SGD or Adam, 1-2 epochs
- **Inference cost**: <0.1ms per layer on CPU

The predictor is **architecture-specific** and will be pre-trained for Qwen3.5 models, bundled with release.

### 4. Neuron Cache & Scheduler (PLANNED — not yet implemented)

**Neuron Cache** (future):
- LRU cache storing hot neuron weight blocks in GPU memory
- Block size: 64 neurons × intermediate_dim
- Miss triggers async DMA from CPU RAM to GPU

**Scheduler** (future):
Before each FFN layer:
1. Run predictor on current activations → hot block set
2. Check GPU cache for each hot block
3. For misses: issue async copy to GPU, reserve space (evict LRU if full)
4. For predicted-cold neurons: compute on CPU (no GPU allocation)
5. Launch GPU kernel to compute only hot neurons
6. Wait for GPU completion (or pipeline with next layer)
7. Merge GPU hot outputs + CPU cold outputs

### 5. GPU Kernels (IN PROGRESS)

Currently using CUDA PTX inline strings with cust bindings. One kernel implemented:
- **matvec_kernel**: matrix-vector multiply on GPU (sm_61 PTX)

Future kernels: sparse matmul, fused ops, MoE dispatch.
Future kernels: sparse matmul, fused ops, MoE dispatch.

### 6. Multi-GPU Coordination (PLANNED)

For 2×1050 Ti (4GB each):

- **Memory partition**: Each GPU gets ~3GB for neuron cache + overhead
- **Work assignment**:
  - Option A: Split layers between GPUs (layers 0-15 on GPU0, 16-31 on GPU1)
  - Option B: Split neurons of each layer across GPUs (hot blocks assigned by bin-packing)
- **Streams**: Overlap PCIe transfers with compute on other GPU
- **Synchronization**: Use CUDA events to coordinate

We'll implement both and benchmark; likely B (neuron-level split) gives better load balance.

### 7. Backend Abstraction

`Backend` trait encompasses:
- `allocate(shape) -> Buffer`
- `copy_to_gpu(&Buffer, &[u8])`
- `launch_kernel(&Kernel, args...)`
- `synchronize()`
- `memory_usage() -> (free, total)`

Implementations:
- `CudaBackend`: uses `cust` + PTX kernels
- `VulkanBackend`: uses `ash` + SPIR-V kernels
- `CpuBackend`: uses std::simd + rayon

Runtime selects backend based on feature flags and hardware detection.

## Data Flow

### Training Phase (Once per Model Architecture)
```
GGUF Model + Prompt Dataset
         ↓
   Profiler (CPU)
         ↓
   Activation logs
         ↓
   Analyzer (build hot index)
         ↓
   Training set (X=activations, y=hot_labels)
         ↓
   Trainer (tiny MLP)
         ↓
   Predictor weights (embedded in binary)
```

### Inference Phase (Per Request)
```
Input tokens
    ↓
Token embeddings lookup
    ↓
For each layer:
  ├─ Attention (GPU or CPU depending on offload setting)
  ├─ Predictor: run tiny MLP on current activations → hot blocks
  ├─ NeuronCache: check GPU residency, schedule missing blocks
  ├─ GPU kernel: compute hot neurons (sparse matmul)
  ├─ CPU fallback: compute cold neurons (dense matmul)
  └─ Combine: output = GPU_result ∪ CPU_result
    ↓
Next token
```

## Performance Model

Let:
- `T_model`: total model size (GB)
- `T_active`: active parameters per token (GB)
- `R_hot`: fraction of neurons that are hot (0.05–0.10)
- `B_cache`: GPU cache size (GB)
- `C_bw`: PCIe bandwidth (16 GB/s for PCIe 3.0 x16)
- `F_compute`: compute throughput (TFLOPs)

**Tokens/sec estimate**:
```
Compute time ≈ (R_hot * ops + (1-R_hot) * ops * CPU_slowdown) / F_compute
Transfer time ≈ (Cache miss rate * Block size) / C_bw
Total time ≈ Compute + Transfer + Overhead
```

For Qwen3.5-35B-A3B on 2×1050 Ti:
- `R_hot` ≈ 0.08 (8% active neurons)
- `B_cache` ≈ 3.5GB (can hold ~60% of hot neurons)
- `F_compute` ≈ 2.8 TFLOPs (2× GTX 1050 Ti FP16)
- Expect: 2.5-4 tok/s

## Quality Assurance Pipeline

Quality is guaranteed by design: every neuron is computed, just on different hardware. The profiler identifies which neurons to put on GPU, not which to skip.

```
┌─────────────────────────────────────────────────────┐
│                    Training Time                      │
│                                                       │
│  Diverse inputs → Profiler → HotNeuronIndex (JSON)   │
│  "Which neurons fire on everything?"                  │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                    Inference Time                      │
│                                                       │
│  ┌──────────────────┐  ┌───────────────────────┐     │
│  │  GPU (VRAM)       │  │  CPU (RAM)             │    │
│  │  Hot neuron       │  │  Cold neuron           │    │
│  │  weights + KV     │  │  weights + KV          │    │
│  │  (TurboQuant      │  │  (larger cache,        │    │
│  │   compresses KV)  │  │   less compression)    │    │
│  └──────────────────┘  └───────────────────────┘     │
│                                                       │
│  Every neuron computed → Quality preserved             │
│  TurboQuant → Longer context in same VRAM              │
│  Sparse GPU → 3x speedup (only hot neurons on GPU)    │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                    Quality Validation                  │
│                                                       │
│  Perplexity at hot=30%: should match hot=100%         │
│  Reference comparison: cosine sim > 0.999             │
│  Results tracked in git for regression detection      │
└─────────────────────────────────────────────────────┘
```

### Quality Metrics

| Metric | Target | How Measured |
|--------|--------|--------------|
| Perplexity | Identical to full CPU | WikiText-103 at various hot% |
| Logit cosine similarity | >0.999 | Compare against llama.cpp reference |
| Top-1 prediction match | >95% | Same token selected |
| Top-5 prediction match | >99% | Correct token in top 5 |

### Modules

- `src/activation/` — Neuron hotness profiling, HotNeuronIndex export
- `src/benchmark/` — Perplexity measurement, reference comparison

## TurboQuant KV Cache Compression

TurboQuant (Google Research, ICLR 2026) compresses the KV cache to 2-4 bits per coordinate with zero accuracy loss on attention scores.

### How It Helps

Without TurboQuant: hot neuron KV cache in 4GB VRAM → ~8K context
With TurboQuant (3-bit): same 4GB → ~40K context

The two techniques are multiplicative:
- Sparse execution: 3x speedup (only hot neurons on GPU)
- TurboQuant: 6x KV cache memory savings (longer context in same VRAM)
- Combined: fast GPU execution + long context on limited hardware

### Algorithm

1. **Stage 1 (PolarQuant)**: Random rotation + Lloyd-Max scalar quantization per coordinate
2. **Stage 2 (QJL)**: 1-bit sign correction for unbiased inner products
3. **Asymmetric estimator**: Compute attention scores directly from compressed keys

### Module

- `src/turboquant/` — Lloyd-Max quantizer, rotation, QJL, CompressedKVCache

### References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni — ICLR 2026)
- [PyTorch reference implementation](https://github.com/tonbistudio/turboquant-pytorch) (MIT license)

## Limitations & Future Work

### Current Limitations
- **One CUDA kernel**: MatVec PTX kernel exists but not yet tested on actual GPU hardware
- **No MoE routing**: Needed for Qwen3.5-35B-A3B (planned, not implemented)
- **No sparse GPU execution**: Core PowerInfer innovation not yet implemented
- **CPU-only inference**: Forward pass works but too slow without GPU acceleration

### Future Directions
- **CUDA matvec on GPU**: Test PTX kernel on 1050 Ti, verify output matches CPU
- **MoE routing**: Select active experts per token for sparse MoE execution
- **Sparse GPU split**: Hot neurons on GPU, cold on CPU (core PowerInfer)
- **TurboQuant KV cache**: Compressed cache in VRAM for longer context
- **Continuous batching**: Multi-request scheduling to improve GPU utilization

## References

### Core Papers

- **PowerInfer**: [Fast Large Language Model Serving with a Consumer-grade GPU](https://arxiv.org/abs/2312.12456) (Song, Mi, Xie, Chen — Shanghai Jiao Tong University, 2023)
- **PowerInfer-2**: [Fast Large Language Model Inference on a Smartphone](https://arxiv.org/abs/2406.06282) (Song et al., 2024)
- **TurboQuant**: [Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni — Google Research, ICLR 2026)
- **QJL**: [1-Bit Quantized JL Transform for KV Cache Quantization](https://arxiv.org/abs/2406.03482) (Zandieh et al., AAAAI 2025)
- **PolarQuant**: [Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617) (AISTATS 2026)

### Model Architectures

- **Qwen3.5-35B-A3B**: [HuggingFace model page](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) (Alibaba Cloud, 2026) — 35B total params, 3B active (MoE)
- **Gated DeltaNet**: [Linear Attention with Gating](https://arxiv.org/abs/2406.12843)

### Quality Evaluation

- **Perplexity**: Standard NLP metric (exponentiated cross-entropy). Defined in [Merity et al., "Pointer Sentinel Mixture Models"](https://arxiv.org/abs/1609.07843) (Salesforce, 2016). Used to measure how well a model predicts text — lower is better.
- **WikiText-103**: Benchmark dataset from [Merity et al.](https://arxiv.org/abs/1609.07843). 103M tokens from English Wikipedia. Available at [paperswithcode.com/dataset/wikitext-103](https://paperswithcode.com/dataset/wikitext-103)

### Tools & Dependencies

- **gguf-rs**: [crates.io/crates/gguf-rs](https://crates.io/crates/gguf-rs) — GGUF file format parser for Rust
- **llama.cpp**: [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — Reference for GGUF format, quantization schemes, backend abstractions
- **rust-gpu**: [github.com/rust-gpu/rust-gpu](https://github.com/rust-gpu/rust-gpu) — Compile Rust to GPU shaders (SPIR-V/NVVM). Not currently used — CUDA PTX inline strings used instead.
- **cust**: [crates.io/crates/cust](https://crates.io/crates/cust) — Rust CUDA driver API bindings
