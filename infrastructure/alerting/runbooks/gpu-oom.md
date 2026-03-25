# Incident: GPU Out of Memory (OOM)

**Severity:** P0 Critical  
**SLO:** Memory usage < 90%  
**Dashboard:** [Grafana - Resources](http://grafana:3000/d/powerinfer-resources)

## Detection

- Alert: `PowerInferGPUMemoryCritical` (>95% for 5 min)
- Logs: `CUDA error: out of memory` or `CUDA error: invalid resource handle`
- Inference failures with OOM errors

## Immediate Response (0-15 min)

1. **Confirm OOM**:
   ```bash
   curl -s 'http://powerinfer:8080/metrics' | grep gpu_memory
   # or if K8s:
   kubectl exec <pod> -- nvidia-smi
   ```

2. **Reduce GPU memory pressure immediately**:
   - **Option A: Restart with fewer GPU layers**
     ```bash
     # Update deployment to use --gpu-layers 16 instead of 28
     kubectl set env deployment/powerinfer POWERINFER_GPU_LAYERS=16
     kubectl rollout restart deployment/powerinfer
     ```
   
   - **Option B: Disable sparse inference** (if using hot index)
     ```bash
     kubectl set env deployment/powerinfer POWERINFER_HOT_INDEX=""
     ```
     This falls back to layer-offload only (less GPU memory efficient but more predictable)

3. **If using multi-GPU**: ensure memory balanced
   ```bash
   # Check per-GPU memory
   kubectl exec <pod> -- nvidia-smi -q -d MEMORY
   ```
   - If one GPU full and others empty: our partitioning may be uneven
   - Temporary fix: use single GPU only (`CUDA_VISIBLE_DEVICES=0`)

## Investigation (15-60 min)

### 1. Model Size vs GPU Capacity

Calculate expected memory:
```
Model size (quantized) × layers_offloaded + KV cache + activations + overhead

Example: Qwen3-8B Q4_K_M ≈ 5.5GB total
- Weights on GPU (28/32 layers): 5.5GB × (28/32) ≈ 4.8GB
- KV cache (batch=1, context=2048): 2 layers × 2 (K/V) × 2048 × 4096 × 2 bytes ≈ 134MB
- Activations: ~500MB
- Overhead: ~200MB
- Total: ~5.6GB
```

If GPU is 6GB, this is cutting it close. Reduce `--gpu-layers`.

### 2. Hot Neuron Cache Overflow

If using sparse inference:
- Check `powerinfer_neuron_cache_hits` vs `powerinfer_neuron_cache_misses`
- High miss rate means predictor is too aggressive, loading many unique hot blocks
- **Mitigation**:
  - Increase neuron cache size (if GPU memory available)
  - Increase predictor threshold (only cache very hot neurons)
  - Reduce `--cache-size` parameter (smaller cache = fewer misses but more CPU fallback)

### 3. Memory Fragmentation

Long-running service may fragment GPU memory. Symptoms:
- Starts fine, OOM after hours of operation
- **Mitigation**:
  - Restart service periodically (daily)
  - Implement memory pool/arena in runtime to reduce fragmentation

### 4. Concurrency Too High

Multiple concurrent requests multiply memory usage:
- Each request needs its own KV cache, activations, intermediate buffers
- Formula: `per_request_memory × max_concurrent`
- Currently `--concurrency 4` means 4× memory for KV cache
- **Mitigation**: Reduce `max_concurrent` or increase queue depth (users wait longer)

### 5. Leaked Memory

Check if memory steadily increases without requests:
```bash
# Monitor over time
watch -n 10 'curl -s http://powerinfer:8080/metrics | grep gpu_memory'
```

- If memory grows even with no traffic: **memory leak** in GPU buffer management
- **Action**: Check `Backend::allocate`/`free` balance; ensure all buffers freed

### 6. Model Accuracy Issue

Sometimes OOM occurs because model file is corrupted or larger than expected:
- Verify GGUF file size matches expected for that quant
- Check model config: `general.parameter_count` vs actual
- Re-download model if corrupted

## Mitigation Strategies

### Short-term
1. **Lower GPU layers** until memory fits comfortably (<85% usage)
2. **Reduce batch size** (if batching enabled) to 1
3. **Limit context length** (`--ctx-size 2048` instead of 8192)
4. **Scale vertically**: Use GPU with more memory (RTX 4090 24GB)
5. **Scale horizontally**: Add more nodes, each with smaller model replica (different approach: sharding)

### Medium-term
- Implement **paged attention** to swap KV cache to CPU when not needed
- Implement **memory-aware scheduler** that automatically adjusts `gpu_layers` based on available VRAM
- Add **model quantization options**: Q4_0 uses less memory than Q4_K_M
- Implement **expert-level offloading for MoE** (only hot experts on GPU)

### Long-term
- **Multi-GPU model parallel**: Split single model across GPUs (tensor parallelism or pipeline)
- **CPU offload for cold weights**: Already planned, but ensure working
- **Memory compression**: Use NVFP4 (4-bit) if available

## Verification

After fix:
```bash
# Check memory usage at steady state
curl -s 'http://prometheus:9090/api/v1/query?query=powerinfer_gpu_memory_usage_bytes' | jq
# Should stay below 85% of total

# Simulate load
hey -z 30s -c 4 -m POST -d '{"prompt":"test","max_tokens":100}' http://localhost:8080/v1/completions
# Monitor memory during test - should not increase monotonically
```

## Prevention

- Set up **capacity alerts** at 80% (warning), 90% (critical)
- **Right-size** GPU memory per model (benchmark memory usage before deployment)
- Implement **circuit breaker** that automatically reduces concurrency when memory >85%
- Regular **load testing** in staging to understand memory profile

---

**Quick Commands**

```bash
# Check GPU memory (inside container/pod)
nvidia-smi

# View metrics
curl -s http://localhost:8080/metrics | grep gpu

# Reduce concurrency
kubectl set env deployment/powerinfer POWERINFER_CONCURRENCY=2

# Disable sparse inference (if enabled)
kubectl set env deployment/powerinfer POWERINFER_HOT_INDEX=""

# Restart
kubectl rollout restart deployment/powerinfer
```
