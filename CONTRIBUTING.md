# Contributing to PowerInfer_x64

Thank you for considering contributing! This document outlines the process for contributing.

## How to Contribute

### Reporting Bugs
- Check existing issues first
- Create a new issue with:
  - Steps to reproduce
  - Expected vs actual behavior
  - Environment (OS, GPU, Rust version, CUDA version)
  - Logs/error messages

### Suggesting Features
- Open an issue with "Feature Request" label
- Describe the use case and expected behavior
- Discuss implementation approach

### Pull Requests
1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Run `cargo fmt` and `cargo clippy -- -D warnings`
5. Ensure tests pass: `cargo test --all-features`
6. Commit with clear, descriptive messages
7. Push and open a PR
8. Wait for review

## Development Setup

### Prerequisites
- Rust nightly-2025-06-23 (use `rustup override set`)
- CUDA 11.8 toolkit (for GPU development) or Vulkan SDK
- Docker (recommended for consistent environment)

### Docker
```bash
docker build -f Dockerfile.cuda -t powerinfer .
docker run --gpus all -v $(pwd):/workspace -it powerinfer
```

### Building
```bash
cargo build --release              # CPU only
cargo build --release --features cuda   # With CUDA
cargo build --release --features vulkan # With Vulkan
cargo build --release --features all    # All features
```

### Testing
```bash
cargo test                          # Unit tests
cargo test --test integration       # Integration tests (requires model)
cargo bench                        # Benchmarks
```

## Code Style
- Follow Rust conventions (`cargo fmt`)
- Use `Result<T>` with `anyhow::Error` for error handling
- Document public APIs with `///` comments
- Add tests for new modules

## Architecture Notes
- Kernel code in `kernels/` (Rust via rust-gpu)
- High-level runtime in `src/`
- GPU backends in `src/runtime/`
- Keep CPU and GPU paths symmetric where possible

## Questions?
Open an issue or reach out in Discussions.
