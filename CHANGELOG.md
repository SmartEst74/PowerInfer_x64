# Changelog

All notable changes to PowerInfer_x64 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Project scaffolding and repository structure
- GGUF parser stub (`src/gguf/`)
- Model configuration framework (`src/model/`)
- Backend abstraction layer (`src/runtime/`)
- Quantization utilities (`src/quant/`)
- Profiler module (`src/profiler/`)
- Predictor module (`src/predictor/`)
- OpenAI-compatible server stub (`src/server/`)
- CLI framework (`src/cli/`)
- GPU kernel placeholders (CUDA + Vulkan)
- Docker development environment
- CI/CD pipeline with GitHub Actions
- SRE infrastructure:
  - Prometheus + Grafana monitoring stack
  - Alertmanager with P0/P1 alert rules
  - 4 incident response runbooks
  - Terraform AWS deployment configuration
  - Kubernetes manifests
  - Grafana dashboards
- Documentation:
  - Architecture document
  - 48-week implementation plan
  - SRE & Infrastructure guide
  - Contributing guidelines
  - Code of Conduct

### Infrastructure
- Docker Compose with full observability stack
- Prometheus scraping configuration
- Alert rules for service health, latency, GPU OOM, throughput
- Terraform modules for VPC, ASG, ALB, IAM
- Kubernetes deployment, service, HPA, PDB manifests

## [0.1.0] - 2026-03-25

### Added
- Initial repository creation
- License (MIT OR Apache-2.0)
