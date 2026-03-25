# PowerInfer_x64 Infrastructure & SRE

This directory contains infrastructure-as-code and SRE configurations for deploying PowerInfer_x64 in production.

## Structure

```
infrastructure/
├── terraform/          # Infrastructure as Code (AWS, GCP, Azure)
│   ├── modules/       # Reusable Terraform modules
│   ├── environments/  # Dev, staging, prod
│   └── scripts/       # Helper scripts
├── monitoring/        # Prometheus configuration
│   ├── prometheus.yml
│   └── rules/
├── alerting/          # Alertmanager configuration
│   └── alertmanager.yml
├── dashboards/        # Grafana dashboard JSONs
│   └── powerinfer.json
├── kubernetes/       # K8s manifests (optional)
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   └── pdb.yaml
└── config/           # Centralized configuration
    └── values.yaml
```

## Quick Start

### Local Development (Docker Compose)

```bash
# Start full stack: PowerInfer + Prometheus + Grafana
docker compose -f deployments/docker-compose.yml up -d

# Access:
# - PowerInfer API: http://localhost:8080
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

### Cloud Deployment (AWS Example)

```bash
cd infrastructure/terraform/environments/prod
terraform init
terraform apply

# Outputs:
# - powerinfer_endpoint = http://<alb-dns>:8080
# - grafana_endpoint = https://<alb-dns>-grafana
```

## SLOs & Monitoring

### Service Level Objectives

| Indicator | Target | Measurement |
|-----------|--------|-------------|
| Availability | 99.9% | HTTP 2xx/3xx responses |
| Latency (p50) | <50ms | API response time |
| Latency (p99) | <500ms | API response time |
| Throughput | >10 req/s | Successful completions |
| Error Rate | <0.1% | 5xx responses |

### Key Metrics

- `powerinfer_inference_requests_total` - Counter of inference requests
- `powerinfer_inference_duration_seconds` - Histogram of request latency
- `powerinfer_tokens_generated_total` - Counter of output tokens
- `powerinfer_gpu_utilization_percent` - GPU compute utilization
- `powerinfer_memory_usage_bytes` - GPU/CPU memory
- `powerinfer_queue_depth` - Pending requests in queue

## Alerting

Critical alerts (P0):
- Instance down (no heartbeat 2min)
- Error rate >1% for 5min
- Latency p99 >2s for 5min
- GPU memory >90% for 10min

Warning alerts (P1):
- Throughput <5 req/s for 10min
- GPU temperature >85°C
- Disk usage >80%

## Incident Response

See `alerting/runbooks/` for playbooks:
- High error rate
- Slow responses
- GPU out of memory
- Model quality degradation

## Scaling Policies

### Horizontal Pod Autoscaler (K8s)

```yaml
minReplicas: 2
maxReplicas: 10
targetCPUUtilizationPercentage: 70
targetMemoryUtilizationPercentage: 80
```

### GPU Node Autoscaling

Custom metrics target:
- GPU utilization >80% → scale up
- GPU utilization <30% → scale down

## Cost Optimization

- Spot instances for non-critical workloads
- GPU scheduling: pack multiple replicas per node
- Auto-scale to zero during off-hours (if applicable)
- Monitor and right-size GPU types

## Compliance & Security

- All traffic TLS (HTTPS)
- Secrets in AWS Secrets Manager / Kubernetes Secrets
- Network policies restrict internal communication
- Regular image vulnerability scanning
- Audit logging to CloudWatch / Stackdriver

## Maintenance

- Nightly: automated integration tests against staging
- Weekly: dependency updates (cargo audit)
- Monthly: capacity planning review
- Quarterly: disaster recovery drill
