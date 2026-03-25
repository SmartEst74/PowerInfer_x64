# Incident: Low Throughput / Underutilization

**Severity:** P1 Warning  
**SLO:** Throughput > 10 requests/sec (adjust based on deployment)  
**Dashboard:** [Grafana - Throughput](http://grafana:3000/d/powerinfer-capacity)

## Detection

- Alert: `PowerInferLowThroughput` (< 5 req/s for 15 min)
- Grafana shows sustained low request rate during expected peak hours
- Business metric: user completion rate down

## Investigation (30-60 min)

### 1. Demand vs Capacity

First, determine if this is **low demand** or **capacity issue**:

```bash
# Check inbound request rate
curl -s 'http://prometheus:9090/api/v1/query?query=sum(rate(powerinfer_inference_requests_total[5m]))'
# Is this low because few clients are connecting, or because requests are queued/waiting?
```

- **Low demand**: Actual request rate is low
  - Check: Load balancer metrics (ALB request count)
  - Possible causes: frontend issue, DNS outage, client-side bug
  - Response: Fix upstream, not PowerInfer

- **High demand but low throughput**: Requests coming in but not being processed fast enough
  - Check: Queue depth (`powerinfer_queue_depth`)
  - If queue depth high → **capacity constrained**
  - If queue depth 0 but throughput low → **per-request latency too high** (see high latency incident)

### 2. Resource Utilization

```bash
# GPU utilization
curl -s 'http://prometheus:9090/api/v1/query?query=powerinfer_gpu_utilization_percent'

# CPU utilization (node-exporter)
curl -s 'http://prometheus:9090/api/v1/query?query=rate(node_cpu_seconds_total{mode!="idle"}[5m])'

# Memory pressure
curl -s 'http://prometheus:9090/api/v1/api/v1/query?query=node_memory_MemAvailable_bytes'
```

**Scenarios**:

| GPU util | CPU util | Diagnosis | Action |
|----------|----------|-----------|--------|
| <30% | <30% | Model not loading on GPU (CPU fallback) | Check `--gpu-layers` config, CUDA errors |
| <30% | >80% | CPU bottleneck | Optimize CPU ops, increase CPU cores |
| 80-100% | <50% | GPU compute bound | Expected for small models; consider larger batch size |
| >90% sustained | | GPU saturated | Scale horizontally (more replicas) |

### 3. Configuration Review

- **Concurrency setting**: If `--concurrency 1`, can only handle 1 request at a time
  ```bash
  kubectl exec <pod> -- ps aux | grep powerinfer
  # Check flags
  ```

- **GPU layers**: Too many layers offloaded may cause memory thrashing, causing lower effective throughput
  - Try reducing `--gpu-layers` by 25% and measure

- **Model size**: If using 35B model on 8GB VRAM with sparse inference, throughput will be low (2-4 tok/s is expected)
  - Consider switching to 8B model if application allows

- **Quantization**: Some quants slower than others
  - Q4_0 fastest, Q6_K slower
  - Profile different quants

### 4. Queueing Behavior

Check if requests are waiting in queue:
```bash
curl -s 'http://prometheus:9090/api/v1/query?query=powerinfer_queue_depth' | jq
```

- Queue depth > `concurrency` means requests waiting
- Cause: `arrival_rate > service_rate`
- Solution: Scale horizontally OR reduce per-request work (shorter context, smaller model)

## Mitigation

### A. Scale Horizontally (Most Common)

Add more replica pods behind load balancer:

```bash
# K8s
kubectl scale deployment powerinfer --replicas=3  # from 1 to 3
# Monitor load distribution; should see throughput increase linearly

# Docker Compose (not built-in scaling; use separate stacks):
docker compose up -d --scale powerinfer=3  # requires v2.4+
```

**Considerations**:
- Each replica needs own GPU (or time-share on same GPU)
- If GPU memory limits 1 replica per node, need more nodes
- Load balancer health checks must pass

### B. Optimize Configuration

1. Increase concurrency (if GPU memory allows):
   ```bash
   kubectl set env deployment/powerinfer POWERINFER_CONCURRENCY=8
   ```

2. Enable batched processing (if supported):
   ```bash
   kubectl set env deployment/powerinfer POWERINFER_BATCH_SIZE=4
   ```

3. Reduce context window (if long contexts not needed):
   ```bash
   kubectl set env deployment/powerinfer POWERINFER_MAX_CONTEXT=1024
   ```

### C. Upgrade Hardware

- More powerful GPU (RTX 4090 > 1050 Ti)
- More GPU memory (allow more layers on GPU)
- Faster CPU for pre/post processing

### D. Model Tiering

- Deploy multiple model sizes:
  - Fast path: 8B for simple queries
  - Slow path: 35B for complex reasoning
  - Route based on request complexity (can be A/B tested)

## Verification

After scaling/optimization:

```bash
# Throughput should increase proportionally to replicas
curl -s 'http://prometheus:9090/api/v1/query?query=sum(rate(powerinfer_inference_requests_total[2m]))'

# GPU utilization should be higher (closer to 80%)
curl -s 'http://prometheus:9090/api/v1/query?query=powerinfer_gpu_utilization_percent'
```

Acceptable:
- Throughput > 10 req/s per replica (or target)
- GPU utilization 60-80% (efficient)
- Queue depth < concurrency

## Long-term Improvements

- **Auto-scaling**: Configure HPA based on request rate or queue depth
  ```yaml
  apiVersion: autoscaling/v2
  kind: HorizontalPodAutoscaler
  spec:
    metrics:
    - type: Pods
      pods:
        metric:
          name: powerinfer_queue_depth
        target:
          type: AverageValue
          averageValue: 50
  ```
- **Async serving**: Use separate worker pool, decouple request acceptance from processing
- **Prioritization**: Priority queues for premium customers
- **Rate limiting**: Prevent abuse and ensure fair sharing

---

**Quick Commands**

```bash
# Current throughput (req/s)
curl -s 'http://prometheus:9090/api/v1/query?query=rate(powerinfer_inference_requests_total[1m])' | jq

# Scale replicas
kubectl scale deployment powerinfer --replicas=5

# Check per-pod throughput
kubectl top pods -l app=powerinfer

# Adjust concurrency
kubectl set env deployment/powerinfer POWERINFER_CONCURRENCY=4
```
