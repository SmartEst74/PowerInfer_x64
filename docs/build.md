# Build Guide

This document provides detailed build instructions for PowerInfer_x64.

## Prerequisites

### System Requirements
- Linux (Ubuntu 22.04+ recommended)
- Rust nightly-2025-06-23
- CUDA 11.8 toolkit (for GPU support)
- 16GB+ RAM (32GB+ recommended for development)
- Docker (for containerized builds)

### Installing Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup install nightly-2025-06-23
rustup override set nightly-2025-06-23
rustup component add rust-src rustc-dev llvm-tools-preview
```

### Installing CUDA Toolkit

Download from NVIDIA's website or use package manager:

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-11-8

# Set environment
export CUDA_PATH=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

### Installing rust-gpu Toolchain

```bash
cargo install --git https://github.com/rust-gpu/rust-gpu.git --rev main rust-gpu
cargo install --git https://github.com/rust-gpu/rust-gpu.git --rev main rustc_codegen_nvvm
```

## Building from Source

### CPU-only Build (No GPU)

```bash
git clone https://github.com/SmartEst74/PowerInfer_x64.git
cd PowerInfer_x64
cargo build --release --features cpu
```

### GPU Build (CUDA)

```bash
git clone https://github.com/SmartEst74/PowerInfer_x64.git
cd PowerInfer_x64
cargo build --release --features cuda
```

### All Features

```bash
cargo build --release --features all
```

## Docker Build

```bash
# Build image
docker build -f Dockerfile.cuda -t powerinfer .

# Run with GPU access
docker run --gpus all -it powerinfer

# Inside container
cargo build --release --features cuda
```

## Verifying Installation

```bash
# Check GPU access
./target/release/powerinfer-cli --help

# Run basic inference
./target/release/powerinfer-cli -m models/Qwen3-8B-Q4_K_M.gguf -p "Hello"
```

## Building for Jetson

For Jetson Orin Nano with Vulkan backend:

```bash
# Cross-compilation (on x86 host)
# Using QEMU or cross-rs for aarch64
# Note: Vulkan backend doesn't require rust-cuda toolchain

# On Jetson device
apt update && apt install -y vulkan-tools libvulkan-dev
cargo build --release --features vulkan
```

## Troubleshooting

### Build Errors

**"LLVM not found"**: Ensure LLVM 7 is installed and `LLVM_CONFIG` is set.

**"NVVM library not found"**: Add `libnvvm.so` to `LD_LIBRARY_PATH` or `PATH`.

**"CUDA toolkit version mismatch"**: Ensure CUDA 11.8 is installed, not 12.x.

### Runtime Errors

**"No GPU detected"**: Ensure CUDA drivers are installed and `nvidia-smi` works.

**"Out of memory"**: Reduce `--gpu-layers` or use CPU backend.

## Performance Optimization

### Build Optimization

```bash
# Profile-guided optimization (requires benchmark data)
cargo build --release --profile release-pgo
```

### Runtime Optimization

```bash
# Increase thread count
./powerinfer-cli --threads 8

# Adjust GPU layers
./powerinfer-cli --gpu-layers 28

# Enable sparse inference
./powerinfer-cli --hot-index model.hot_index.bin
```

## CI/CD

The project uses GitHub Actions for continuous integration. See `.github/workflows/ci.yml` for details.

## Next Steps

- [Architecture Guide](architecture.md) - System design
- [Performance Tuning](performance.md) - Optimization guide
- [Troubleshooting](troubleshooting.md) - Common issues
