# Terraform Infrastructure for PowerInfer_x64

This directory contains Terraform configurations for deploying PowerInfer_x64 to cloud providers.

## Modules

### `modules/network`
VPC with public/private subnets, NAT gateway, security groups.

### `modules/compute`
GPU-enabled compute (EC2 G5/G6 instances or equivalent).

### `modules/storage`
S3 bucket for model storage, EBS volumes for local cache.

### `modules/monitoring`
CloudWatch/Prometheus/Grafana stack.

## Environments

- **development**: Single-node, t3.medium + mock GPU (or no GPU)
- **staging**: Production-like but smaller scale (1-2 GPU nodes)
- **production**: Multi-AZ, auto-scaling, load balanced

## Usage

```bash
# Initialize
terraform init

# Plan
terraform plan -var="environment=prod"

# Apply
terraform apply -var="environment=prod"

# Destroy
terraform destroy
```

## Variables

See `variables.tf` for all configurable parameters.

### Required

- `aws_region` (or equivalent for GCP/Azure)
- `model_s3_uri` (e.g., `s3://my-bucket/models/Qwen3-8B-Q4_K_M.gguf`)
- `instance_type` (e.g., `g5.xlarge` for single A10)

### Optional

- `min_instances` (default: 1)
- `max_instances` (default: 5)
- `enable_monitoring` (default: true)

## Cost Estimates (AWS us-east-1)

- **Development** (g4dn.xlarge, 1 instance): ~$0.65/hr = ~$470/month
- **Staging** (g5.xlarge, 2 instances): ~$1.20/hr = ~$870/month
- **Production** (g5.2xlarge, auto-scale 2-5): ~$2.40-6.00/hr = ~$1800-4500/month

Prices vary by region and spot instance usage.
