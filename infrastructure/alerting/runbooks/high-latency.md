# Incident: High Latency (Slow Responses)

**Severity:** P0 Critical if p99 >2s, P1 Warning if p95 >500ms  
**SLO:** p99 latency < 500ms, p95 < 200ms  
**Dashboard:** [Grafana - Performance](http://grafana:3000/d/powerinfer-performance)

## Detection

- Alert: `PowerInferHighLatency` (p99 > 2s for 5 min)
- User complaints about slow responses
- Monitoring shows latency spike

## Immediate Response (0-15 min)

1. **Check current metrics**:
   ```bash
   # From Prometheus: p50, p95, p99 latency
   curl -s 'http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,rate(powerinfer_inference_duration_seconds_bucket[5m]))'
   ```

2. **Identify slow requests**:
   - Are all requests slow or just specific models/endpoints?
   - Check if `/v1/completions` vs `/v1/chat/completions` differ

3. **Quick scale test**:
   ```bash
   kubectl scale deployment powerinfer --replicas=3
   # Or increase Docker Compose replicas
   ```
   - If latency drops: **capacity issue** → keep scaled up, investigate queue
   - If latency persists: **code/configuration issue** → continue investigation

4. **Enable debug logging** temporarily:
   ```bash
   kubectl set env deployment/powerinfer POWERINFER_LOG_LEVEL=debug
   ```

## Investigation (15-60 min)

### 1. GPU Utilization

```bash
# Check if GPU is bottlenecked
curl -s 'http://powerinfer:8080/metrics' | grep gpu_utilization
```

- **GPU util < 30%**: Model likely CPU-bound or I/O bound
  - Check: disk read speed for model weights
  - Check: CPU utilization (maybe insufficient CPU cores for pre/post processing)
  - Action: Increase `--threads` or optimize CPU-bound ops

- **GPU util 80-100%**: GPU is the bottleneck
  - Typical: limited by compute throughput (TFLOPs)
  - If using sparse inference: check cache hit rate
  - Action: Reduce batch size, reduce model size, or upgrade GPU

### 2. Queue Depth

```bash
curl -s 'http://prometheus:9090/api/v1/query?query=powerinfer_queue_depth'
```

- **Queue depth > 100**: backlog of requests
  - Cause: throughput < request rate
  - Action: Scale horizontally, optimize throughput, add rate limiting

- **Queue depth 0 but high latency**: per-request latency issue
  - Check individual request trace (if distributed tracing enabled)
  - Look for slow KV cache operations, slow dequantization

### 3. Model Configuration

- **Context length**: Very long contexts (32K+) cause KV cache overhead
  - Check: `prompt_tokens` in request
  - Mitigation: Implement sliding window attention or limit max context

- **Quantization**: Some quant formats slower than others (Q4_K_M vs Q8_0)
  - Benchmark different quants on same hardware
  - Use faster quant if quality acceptable

- **Sparse inference**: Hot neuron predictor overhead
  - Predictor should be <0.1ms; check if misbehaving
  - Temporarily disable: remove `--hot-index` flag (layer-offload only)

### 4. Resource Contentions

- **PCIe bandwidth** (multi-GPU setups): 
  - CPU↔GPU or GPU↔GPU transfers saturating PCIe 3.0 x16 (~16 GB/s)
  - Check: `nvidia-smi pmon` (if accessible)
  - Mitigation: Batch transfers, reduce model layer offloading

- **Memory bandwidth** (CPU RAM ↔ CPU cache):
  - If CPU-bound: likely memory bandwidth limited
  - Use numactl to bind to single NUMA node if on multi-socket

- **Disk I/O** (model loading from disk on cold start):
  - Ensure model weights are cached in RAM (mlock or vmtouch)

## Mitigation Strategies (pick based on cause)

### A. Reduce Per-Request Work
- Limit `max_tokens` to lower values
- Use smaller model variant (e.g., 8B instead of 35B)
- Disable streaming if client doesn't need it (reduces overhead)

### B. Increase Parallelism
- Increase concurrency (`--concurrency` flag)
- Scale horizontally (more replicas behind load balancer)
- Use GPU batching if supported (merge multiple requests)

### C. Optimize Configuration
- For Qwen3.5: try `--gpu-layers` lower value if GPU memory pressure
- Increase CPU threads (`--threads`) if CPU-bound
- Use different quantization (Q4_0 fastest, Q8_0 highest quality)

### D. Cache Frequently Used Prompts
- If same prompts repeated (e.g., system prompts), cache KV state
- Implement prompt template caching

### E. Upgrade Infrastructure
- Faster GPU (RTX 4090 > 1050 Ti)
- More GPU memory (12GB+)
- PCIe 4.0/5.0 for bandwidth-bound scenarios

## Verification

After applying fix:
```bash
# Wait 2-5 min for metrics to stabilize
# Check p95/p99 latency
curl -s 'http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(powerinfer_inference_duration_seconds_bucket[5m]))' | jq '.data.result[0].value[1]'
curl -s 'http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,rate(powerinfer_inference_duration_seconds_bucket[5m]))' | jq '.data.result[0].value[1]'
```

Acceptable:
- p95 < 200ms
- p99 < 500ms

## Communication

- Update incident channel when latency improves
- Document root cause and action taken
- Consider adding dashboard panel for long-term tracking

## Prevention

- Implement auto-scaling based on latency SLOs
- Add request timeout limits (e.g., max 10s per request)
- Implement load shedding when queue > threshold
- Profile production traces monthly to catch regressions

---

**Quick Commands**

```bash
# Check latency SLOs
curl -s 'http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(powerinfer_inference_duration_seconds_bucket[5m]))' | jq

# View top slow requests (if tracing enabled)
jaeger-ui query --lookback 1h --operation Inference

# Scale horizontally
kubectl scale deployment powerinfer --replicas=5

# Restart with debug logging
kubectl set env deployment/powerinfer POWERINFER_LOG_LEVEL=debug
```
