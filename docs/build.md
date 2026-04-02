# Build Guide

This guide reflects the current repository state as of 2026-04-02.

## Toolchain

- Default toolchain: stable Rust from `rust-toolchain.toml`
- Optional features: `cuda`, `vulkan`, `server`, `profiling`
- `nightly-2025-06-23` is only needed when experimenting with future `rust-gpu` kernel compilation

## Requirements

### CPU-only development

- Linux
- `rustup`
- enough RAM and disk for the target GGUF file

### CUDA work

- NVIDIA driver
- CUDA toolkit installed and visible to the build
- build with `--features cuda`

### Vulkan work

- Vulkan loader and development headers
- build with `--features vulkan`

## Build Commands

### Release build

```bash
cargo build --release
```

### CUDA build

```bash
cargo build --release --features cuda
```

### Vulkan build

```bash
cargo build --release --features vulkan
```

### Server build

```bash
cargo build --release --features server
```

### Profiling build

```bash
cargo build --release --features profiling
```

## Verification Commands

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo run --release --bin gguf_dump -- /path/to/model.gguf
cargo run --release --bin real_test -- /path/to/model.gguf
```

## Current CLI Entry Points

### Generate text

`powerinfer-cli` is subcommand-based.

```bash
cargo run --release --bin powerinfer-cli -- generate \
    --model /path/to/model.gguf \
    --prompt "The capital of France is" \
    -n 1
```

### Start the HTTP server

```bash
cargo run --release --features server --bin powerinfer-serve -- /path/to/model.gguf
```

Current limitation: the server returns real model output for basic requests, but it is still greedy-only and does not implement streaming.

### Run the profiler scaffold

```bash
cargo run --release --features profiling --bin powerinfer-profile -- \
    --model /path/to/model.gguf \
    --output profile.jsonl
```

Current limitation: the profiler currently performs model analysis only. It does not generate a finished hot-neuron profile or index.

### Inspect GGUF metadata

```bash
cargo run --release --bin gguf_dump -- /path/to/model.gguf
```

### Run the full real-model validation path

```bash
cargo run --release --bin real_test -- /path/to/model.gguf
```

## Docker

A CUDA-oriented development container is available.

```bash
docker build -f Dockerfile.cuda -t powerinfer .
docker run --gpus all -it -v $(pwd):/workspace powerinfer
```

Inside the container, use the same `cargo` commands shown above.

## Troubleshooting

### `warning: powerinfer@0.1.0: No GPU backend selected (CPU-only mode)`

This is informational. It means you built or ran a CPU-only path.

### Slow performance in `cargo run`

Use `--release` for any benchmark or throughput claim. The dev profile is dramatically slower.

### Helper binary complains about a missing model path

The helper tools now expect an explicit GGUF path. Pass `/path/to/model.gguf` directly.

### `cargo check` resolves the wrong parent project

This repo has a known local-environment quirk documented in `AGENTS.md`: stale `/home/jon/Cargo.toml` and `/home/jon/src/` can interfere with cargo commands. Follow the documented temporary rename workaround when that happens.
