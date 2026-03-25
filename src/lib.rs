//! PowerInfer_x64: Neuron-level sparse LLM inference in pure Rust
//!
//! This crate provides:
//! - GGUF model loading
//! - CPU and GPU (CUDA/Vulkan) backends
//! - Activation profiling and hot neuron prediction
//! - Sparse inference with predictive neuron caching
//! - OpenAI-compatible HTTP server
//!
//! ## Quick Example
//!
//! ```no_run
//! use powerinfer::{gguf::GgufFile, model::InferenceContext, runtime::Backend};
//!
//! # fn main() -> anyhow::Result<()> {
//! let gguf = GgufFile::open("models/Qwen3-8B-Q4_K_M.gguf")?;
//! let mut ctx = InferenceContext::from_gguf(gguf, Backend::cpu())?;
//!
//! let output = ctx.generate("Hello, world!", 100)?;
//! println!("{}", output);
//! # Ok(())
//! # }
//! ```

pub mod gguf;
pub mod model;
pub mod quant;
pub mod runtime;
pub mod profiler;
pub mod predictor;
pub mod server;
pub mod cli;
pub mod tokenizer;

// Re-exports
pub use gguf::GgufFile;
pub use model::{InferenceContext, ModelConfig};
pub use runtime::Backend;

/// Result type alias with anyhow::Error
pub type Result<T> = std::result::Result<T, anyhow::Error>;
