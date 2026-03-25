# Incident: Service Down / High Error Rate

**Severity:** P0 Critical  
**SLO:** Availability < 99.9%  
**Dashboard:** [Grafana - Service Overview](http://grafana:3000/d/powerinfer-service)

## Detection

- Alert: `PowerInferDown` or `PowerInferHighErrorRate`
- Notification received via Slack/PagerDuty

## Immediate Response (0-15 min)

1. **Acknowledge** the incident in the incident channel
2. **Check dashboard**:
   ```bash
   # Quick status check
   curl -s http://powerinfer:8080/health
   ```
3. **Gather initial data**:
   - Current request rate (Prometheus: `rate(powerinfer_inference_requests_total[5m])`)
   - Error breakdown by status code
   - Recent deployments (`git log --oneline -10`)
4. **Determine if rollback needed**:
   - If incident started after recent deploy (<1 hour): consider rollback
   - Execute: `./scripts/rollback.sh <commit-hash-before-incident>`

## Investigation (15-60 min)

1. **Examine logs**:
   ```bash
   # Jail logs for errors
   docker compose logs -f --tail=100 powerinfer | grep -i error
   # Or if using K8s:
   kubectl logs -f deployment/powerinfer --tail=100 | grep -i error
   ```

2. **Check resource utilization**:
   - GPU memory: `powerinfer_gpu_memory_usage_bytes`
   - Queue depth: `powerinfer_queue_depth`
   - CPU/Memory of host

3. **Validate model loading**:
   - Check if model file exists and is readable
   - Verify GGUF integrity
   - Check hot index loading (if using sparse inference)

4. **Test direct inference**:
   ```bash
   # Simple cURL test
   curl -X POST http://localhost:8080/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"prompt":"test","max_tokens":10}' \
     -v
   ```

## Mitigation Options

### Option A: Restart Service (Quick Fix)
```bash
docker compose restart powerinfer
# or K8s: kubectl rollout restart deployment/powerinfer
```
- **Effect**: Clears state, reloads model
- **Risk**: May cause cold starts; temporary

### Option B: Scale Up Instances
```bash
# Docker Compose: increase replicas in docker-compose.override.yml
# K8s:
kubectl scale deployment powerinfer --replicas=5
```
- **Effect**: Distributes load, masks underlying issue
- **Use**: If issue is capacity-related

### Option C: Disable Problematic Features
- If using hot index sparse inference, try CPU-only or layer-offload only:
  ```bash
  ./powerinfer-cli --gpu-layers 0  # Force CPU
  ```
- Or reduce concurrency

### Option D: Rollback to Previous Version
```bash
./scripts/rollback.sh <previous-stable-commit>
```

### Option E: Emergency Maintenance Mode
- Return HTTP 503 with Retry-After for all requests
- Allows investigation without user impact
- Update load balancer health check to fail

## Communication

- Update incident channel every 30 minutes with status
- Post estimated time to resolution (ETA)
- Escalate to engineering leadership if not resolved in 1 hour

## Post-Incident (After Recovery)

1. **Fill out postmortem** using [template](https://github.com/SmartEst74/PowerInfer_x64/blob/main/docs/postmortem-template.md)
2. **Identify root cause** (5 Whys)
3. **Action items** to prevent recurrence:
   - Better monitoring?
   - Circuit breaker?
   - Improved error handling?
4. **Share findings** in team meeting

## Prevention

- Implement circuit breaker pattern for model loading failures
- Add model health check at startup
- Set up canary deployments with 5% traffic
- Increase test coverage for error paths

---

**Quick Commands Reference**

```bash
# Health check
curl http://localhost:8080/health

# Metrics endpoint
curl http://localhost:8080/metrics

# View recent logs (Docker)
docker compose logs -f powerinfer

# Restart service
docker compose restart powerinfer

# Scale (K8s)
kubectl scale deployment powerinfer --replicas=3

# Rollback (K8s)
kubectl rollout undo deployment/powerinfer
```
