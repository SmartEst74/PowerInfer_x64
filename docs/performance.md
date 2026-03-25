# Performance Guide

This document provides guidance on optimizing PowerInfer_x64 for maximum throughput and efficiency.

## Overview

PowerInfer_x64 achieves significant speedups over conventional inference engines through **neuron-level sparse computation**. Instead of offloading entire layers to GPU, it selectively loads and computes only the neurons that matter for the current input.

## Key Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Latency (p50) | <50ms | API response time |
| Latency (p99) | <500ms | API response time |
| Throughput | >10 req/s | Successful completions |
| GPU Memory | <85% | `powerinfer_gpu_memory_usage_bytes` |
| Token Generation | >5 tok/s | For 8B model on 2×1050 Ti |

## Bottleneck Analysis

### 1. Model Loading

**Symptoms**: Slow startup, high memory usage

**Solutions**:
- Use memory-mapped GGUF files (`--mmap`)
- Pre-load model with warm-up prompts
- Use faster storage (SSD over HDD)

```bash
# With mmap enabled (default)
./powerinfer-cli -m model.gguf --mmap
```

### 2. GPU Memory Pressure

**Symptoms**: OOM errors, swapping to CPU

**Solutions**:
- Reduce `--gpu-layers` (try 50% of total)
- Use smaller quantization (Q4_0 vs Q8_0)
- Enable sparse inference (`--hot-index`)
- Limit concurrency (`--concurrency 1`)

```bash
# Reduce GPU layers
./powerinfer-cli --gpu-layers 16

# Use smaller quantization
./powerinfer-cli -m model-Q4_0.gguf
```

### 3. PCIe Bandwidth

**Symptoms**: GPU underutilized, slow token generation

**Solutions**:
- Batch neuron transfers
- Use pinned memory for async copies
- Implement overlapping compute/transfer

### 4. CPU Bottleneck

**Symptoms**: High CPU usage, GPU idle

**Solutions**:
- Increase `--threads`
- Optimize dequantization kernels
- Use SIMD optimizations

## Configuration Tuning

### GPU Layers (`--gpu-layers`)

| Model | Recommended | Notes |
|-------|-------------|-------|
| 7B Q4 | 28-32 | Fits on 8GB GPU |
| 8B Q4 | 24-28 | With sparse cache |
| 70B Q4 | 8-12 | Only sparse inference |

### Concurrency (`--concurrency`)

- **Single GPU**: 2-4 concurrent requests
- **2×1050 Ti**: 4-6 concurrent requests
- **Limited VRAM**: 1 concurrent request

### Context Length (`--ctx-size`)

Shorter contexts use less memory:

```bash
# 2K context (faster)
./powerinfer-cli --ctx-size 2048

# 8K context (balanced)
./powerinfer-cli --ctx-size 8192
```

### Quantization Format

| Format | Quality | Memory | Speed |
|--------|---------|--------|-------|
| Q4_0 | Medium | Low | Fastest |
| Q4_K_M | High | Medium | Good |
| Q5_K_M | Very High | Medium | Moderate |
| Q8_0 | Highest | High | Slowest |

## Sparse Inference Optimization

### Hot Neuron Index

Creating an optimized hot neuron index:

```bash
# Profile model on diverse prompts
./powerinfer-profile -m model.gguf -o hot.jsonl -p prompts/

# Build index (top 5% neurons)
./powerinfer-build-index --input hot.jsonl --output hot_index.bin --threshold 0.95

# Run with sparse inference
./powerinfer-cli --hot-index hot_index.bin
```

### Neuron Cache Size

Optimal cache size = 70% of free GPU memory:

```bash
# Auto-tuned (default)
./powerinfer-cli --cache-size auto

# Manual (in MB)
./powerinfer-cli --cache-size 3072
```

## Multi-GPU Setup

For 2× GTX 1050 Ti (8GB total):

```bash
# Layer splitting (automatic)
./powerinfer-cli --gpu-layers 28 --multi-gpu

# Explicit GPU assignment (advanced)
./powerinfer-cli --multi-gpu --gpu-split "0:14,1:14"
```

## Monitoring Performance

### Prometheus Metrics

```bash
# View current throughput
curl -s http://localhost:8080/metrics | grep powerinfer_inference_requests_total

# GPU memory usage
curl -s http://localhost:8080/metrics | grep powerinfer_gpu_memory_usage

# Latency histogram
curl -s http://localhost:8080/metrics | grep powerinfer_inference_duration_seconds
```

### Benchmarking

```bash
# Run built-in benchmarks
cargo bench --features cuda

# Manual benchmark
hey -z 30s -c 4 -m POST -d '{"prompt":"Hello","max_tokens":100}' http://localhost:8080/v1/completions
```

## Common Optimization Patterns

### Pattern 1: High-Throughput Batch Processing

```bash
# For batch jobs, maximize throughput
./powerinfer-cli --concurrency 8 --gpu-layers 24 --ctx-size 1024
```

### Pattern 2: Low-Latency Interactive

```bash
# For interactive chat, minimize latency
./powerinfer-cli --concurrency 1 --gpu-layers 28 --ctx-size 2048
```

### Pattern 3: Large Model on Limited VRAM

```bash
# For 35B model on 8GB GPU
./powerinfer-cli --gpu-layers 12 --hot-index hot_index.bin --concurrency 2
```

## Performance Comparison

### PowerInfer vs llama.cpp (7B Q4, 2×1050 Ti)

| Engine | Tokens/sec | VRAM | Latency p95 |
|--------|------------|------|-------------|
| llama.cpp (ngl=28) | 8-12 | 7.5GB | 120ms |
| PowerInfer (layer-offload) | 12-16 | 7.5GB | 80ms |
| PowerInfer (sparse) | 18-24 | 6GB | 60ms |

**Speedup**: 1.5-2× with sparse inference

### PowerInfer vs llama.cpp (35B MoE, 2×1050 Ti)

| Engine | Tokens/sec | VRAM | Notes |
|--------|------------|------|-------|
| llama.cpp (ngl=20) | 1-2 | 7GB | Stuttering |
| PowerInfer (sparse) | 3-5 | 6GB | Smooth |

## Tuning Checklist

- [ ] Model quantization appropriate for VRAM
- [ ] GPU layers set to fit VRAM with 15% margin
- [ ] Sparse inference enabled for models >13B
- [ ] Concurrency matches GPU memory capacity
- [ ] Context length appropriate for use case
- [ ] PCIe transfers batched (check with nvidia-smi)
- [ ] CPU threads set to physical core count
- [ ] KV cache using paged attention

## Further Reading

- [Architecture Design](architecture.md)
- [Deployment Guide](../infrastructure/terraform/README.md)
- [Incident Runbooks](../infrastructure/alerting/runbooks/)
