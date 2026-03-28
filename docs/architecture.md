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
│  Model Loader │ Scheduler │ Predictor │ Neuron Cache │ etc │
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
│           GPU Kernels (Rust via rust-gpu)                  │
├─────────────────────────────────────────────────────────────┤
│  SparseMatMul │ FusedOps │ MoERouter │ DeltaNet          │
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

### 4. Neuron Cache & Scheduler

**Neuron Cache**:
- LRU cache storing hot neuron weight blocks in GPU memory
- Block size: 64 neurons × intermediate_dim (e.g., 64×1024 = 65K values = 128KB f16)
- Total cache size: 3-4GB (fits in 4GB GPU after overhead)
- Miss triggers async DMA from CPU RAM to GPU

**Scheduler**:
Before each FFN layer:
1. Run predictor on current activations → hot block set
2. Check GPU cache for each hot block
3. For misses: issue async copy to GPU, reserve space (evict LRU if full)
4. For predicted-cold neurons: compute on CPU (no GPU allocation)
5. Launch GPU kernel to compute only hot neurons
6. Wait for GPU completion (or pipeline with next layer)
7. Merge GPU hot outputs + CPU cold outputs

### 5. GPU Kernels

Written in Rust using `rust-gpu` (compiled to PTX or SPIR-V).

#### SparseMatMul
```rust
#[kernel]
fn sparse_matmul_hot(
    hot_indices: &[u32],        // Indices of active output neurons
    weights: &[f16],            // Weight matrix [n_out, n_in] (full in RAM)
    activations: &[f16],        // Input vector [n_in]
    output: &mut [f16],         // Output vector [n_out], only hot written
    n_out: usize,
    n_in: usize,
    n_hot: usize,
) { /* ... */ }
```

Only computes rows corresponding to hot neuron indices. Cold rows skipped entirely.

#### FusedOps
Fuses multiple operations to reduce memory traffic:
```
input → RMSNorm → RoPE → SwiGLU → MatMul → output
```
All in one kernel, reading weights from neuron cache.

#### MoERouter
For MoE layers:
- Compute token embeddings → expert affinities
- Select top-8 experts (gating + shared)
- Scatter token to expert buffers
- Launch expert kernels (each expert is dense FFN)

#### DeltaNet
Gated DeltaNet recurrent attention for Qwen3.5:
- Maintain state vector per sequence
- Update: `state = A * state + B * input`
- Output: `y = C * state + D * input`
- Chunked parallelism over sequence dimension

All kernels designed for **CUDA** (sm_61, sm_75, sm_87) and **Vulkan** (compute shader).

### 6. Multi-GPU Coordination

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

- Paper: [arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874)
- PyTorch reference: [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)

## Limitations & Future Work

### Current Limitations
- **No pure Rust CUDA kernels yet**: rust-gpu is experimental; we'll initially use C++ kernels via llama.cpp as fallback
- **Predictor training required per architecture**: Manual step for new model families
- **DeltaNet chunking not fully optimized**: Autoregressive nature limits parallelism
- **No flash attention**: KV cache still full size; could add sliding window

### Future Directions
- **Online predictor adaptation**: Fine-tune predictor during inference based on observed hotness
- **Speculative decoding**: Use MTP-like module for 2× speed
- **Kernel fusion in Rust**: Move all kernels to pure rust-gpu (long-term)
- **Quantization-aware sparsity**: Different hot thresholds per quant level
- **Continuous batching**: Multi-request scheduling to improve GPU utilization

## References

- [Original PowerInfer](https://github.com/tiiny-ai/powerinfer) - Neuron-level sparsity concept
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF, quantization, backend abstractions
- [Qwen3.5 Technical Report](https://arxiv.org/abs/2504.1006) - Architecture details
- [Gated DeltaNet](https://arxiv.org/abs/2406.12843) - Linear attention mechanism
- [rust-gpu](https://github.com/rust-gpu/rust-gpu) - Rust on GPUs
