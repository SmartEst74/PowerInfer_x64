# AGENTS.md

## Understand Current State First

Before doing any work, run these commands to understand the project:

```bash
# 1. Verify code health
cargo test
cargo clippy --all-targets -- -D warnings

# 2. Read honest quality assessment
cat QUALITY.md

# 3. Check open issues (what needs doing)
gh issue list --state open

# 4. Check recent commits (what was just done)
git log --oneline -10

# 5. Check if real models load
cargo run --bin gguf_dump -- /home/jon/models/llama-cache/Arch-Agent-3B.Q8_0.gguf
```

These commands tell you: what works, what's broken, what's planned, and what was just done. Do not proceed with work until you understand the output of all five.

## Build & Test Commands

```bash
# Build
cargo build

# Run all tests (must pass before committing)
cargo test

# Run with server feature
cargo test --features server

# Clippy (must pass with -D warnings)
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features server -- -D warnings

# Format check
cargo fmt --all --check

# Test against real model
cargo run --bin real_test -- /home/jon/models/llama-cache/Arch-Agent-3B.Q8_0.gguf

# Dump GGUF metadata
cargo run --bin gguf_dump -- /path/to/model.gguf
```

## Tech Stack

- Rust nightly-2025-06-23 (for rust-gpu)
- `gguf-rs` 0.1.7 (GGUF parsing)
- `half` 2.4 (f16 support)
- `anyhow` 1.0 (error handling)
- Axum 0.7 + Tokio (HTTP server, optional)
- Prometheus 0.13 (metrics, optional)

## Project Structure

- `src/quant/mod.rs` — Dequantization kernels (Q4_0, Q8_0, Q5_0, Q5_1, F32, F16). You READ and WRITE here.
- `src/ops.rs` — Tensor ops (RMSNorm, RoPE, attention, SwiGLU, matvec). You READ and WRITE here.
- `src/gguf/mod.rs` — GGUF parser, model config. You READ and WRITE here.
- `src/model/mod.rs` — Forward pass, InferenceContext. You READ and WRITE here.
- `src/weights.rs` — Weight loader (GGUF → f32). You READ and WRITE here.
- `src/tokenizer.rs` — BPE tokenizer. You READ and WRITE here.
- `src/turboquant/mod.rs` — KV cache compression (TurboQuant). You READ here.
- `src/activation/mod.rs` — Neuron hotness profiler. You READ here.
- `src/benchmark/mod.rs` — Perplexity, reference comparison. You READ here.
- `src/server/mod.rs` — HTTP server. You READ here.
- `src/runtime/mod.rs` — Backend trait, CPU/CUDA/Vulkan. You READ here.
- `src/metrics.rs` — Prometheus metrics. You READ here.
- `src/cli.rs` — CLI entry point. You READ here.
- `src/profiler.rs` — Profiler binary entry point. You READ here.
- `src/bin/` — Binary entry points (real_test, gguf_dump, powerinfer-serve).
- `tests/` — Smoke, soak, integration tests. You WRITE here.
- `docs/` — Architecture, build, performance docs. You READ here.
- `AGENTS.md` — This file. You READ here.
- `QUALITY.md` — Quality baseline and honest gaps. You READ here.
- `PLAN.md` — 48-week implementation plan. You READ here.

## Code Style

- Use `anyhow::Result` throughout, not `std::result::Result`
- Use `#[allow(non_camel_case_types)]` for GGML type enums
- Use `crate::` in lib modules, `powerinfer::` in binary files
- Use architecture-agnostic `get_config()` for GGUF keys (supports qwen2, qwen3, llama)
- Inline format args: `format!("{x}")` not `format!("{}", x)`
- Use `.div_ceil()` not `(x + n - 1) / n`

## Boundaries

### Always Do

- Run `cargo clippy --all-targets -- -D warnings` before committing
- Run `cargo test` before committing
- Update `QUALITY.md` with results after real model tests
- Cite sources for external references (arXiv papers, GitHub repos)
- Test against `/home/jon/models/llama-cache/Arch-Agent-3B.Q8_0.gguf` for real model validation

### Ask First Before

- Adding new dependencies to Cargo.toml
- Changing the GGUF parser API (affects all modules)
- Modifying the forward pass architecture
- Adding new binary entry points

### Never Do

- Hardcode `llama.` prefix for GGUF keys — use `get_config()` which supports any architecture
- Use `crate::` in binary files — use `powerinfer::`
- Claim "works" without a real model test
- Write tests only on synthetic data
- Add features without an open GitHub issue
- Use `cargo check` without hiding the stale parent files (see Important Notes)

## Git Workflow

- Commit message format: imperative mood, explain WHY not WHAT
- Example: `Fix GGUF config parser to support architecture-specific prefixes`
- Run `cargo clippy` and `cargo test` before every commit

## Available Models (on disk)

```
/home/jon/models/llama-cache/
  Arch-Agent-3B.Q8_0.gguf              3.1GB  Q8_0   Qwen2 arch ✅ WORKS
  Qwen3-4B-Instruct-2507-Q4_K_M.gguf  2.4GB  Q4_K_M Qwen3 arch ❌ NEEDS Q5_K
  gemma-3-4b-it-Q4_K_M.gguf           2.4GB  Q4_K_M Gemma arch ❌ NEEDS Q4_K_M
  BitAgent-Bounty-8B.Q4_K_M.gguf      4.6GB  Q4_K_M ❌ NEEDS Q4_K_M
  Salesforce.Llama-xLAM-2-8b-fc-r.Q4_K_M.gguf 4.6GB Q4_K_M ❌ NEEDS Q4_K_M
```

## Verified References

- PowerInfer: https://arxiv.org/abs/2312.12456
- PowerInfer-2: https://arxiv.org/abs/2406.06282
- TurboQuant: https://arxiv.org/abs/2504.19874
- Qwen3.5-35B-A3B: https://huggingface.co/Qwen/Qwen3.5-35B-A3B
- gguf-rs: https://crates.io/crates/gguf-rs
- llama.cpp (reference for quant formats): https://github.com/ggml-org/llama.cpp

## Important Notes

- Stale `/home/jon/Cargo.toml` and `/home/jon/src/` interfere with `cargo check`. Hide them before building:
  ```bash
  mv /home/jon/Cargo.toml /home/jon/Cargo.toml.bak
  mv /home/jon/src /home/jon/src.bak
  cargo check
  mv /home/jon/Cargo.toml.bak /home/jon/Cargo.toml
  mv /home/jon/src.bak /home/jon/src
  ```
- The project uses `nightly-2025-06-23` toolchain (for rust-gpu)

## Quality Gates (L5 Architect + Uncle Bob Standards)

Before claiming any work is done, verify ALL of the following:

### Uncle Bob (Clean Code)
- [ ] Tests verify USER-FACING BEHAVIOR, not just internal math
- [ ] Every public function has a test that would fail if the function broke
- [ ] No dead code (clippy -D warnings must pass)
- [ ] Functions do one thing (no 100-line functions)
- [ ] Test names describe WHAT the user gets, not HOW it works

### L5 Architect (Production Quality)
- [ ] Integration test against a REAL model file (not synthetic data)
- [ ] Performance benchmarked (tok/s tracked in QUALITY.md)
- [ ] QUALITY.md updated with honest results (what works AND what doesn't)
- [ ] Issues tracked: completed work closed, new work has an open issue
- [ ] No claim of "working" without end-to-end proof

### The "Would Uncle Bob Approve?" Checklist
Before committing, ask:
1. Could a new developer understand this code in 10 minutes? → If no, refactor
2. Does the test PROVE the feature works for the user? → If no, add acceptance test
3. Would I be embarrassed showing this to a senior engineer? → If yes, fix it
4. Is there an open issue for what I'm about to do next? → If no, create one
