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
//! use powerinfer::model::InferenceContext;
//! use powerinfer::runtime::BackendFactory;
//!
//! # fn main() -> anyhow::Result<()> {
//! let mut ctx = InferenceContext::from_gguf("models/Qwen3-8B-Q4_K_M.gguf", BackendFactory::cpu())?;
//!
//! let output = ctx.generate("Hello, world!", 100)?;
//! println!("{}", output);
//! # Ok(())
//! # }
//! ```

pub mod activation;
pub mod benchmark;
pub mod gguf;
pub mod model;
pub mod ops;
pub mod quant;
pub mod runtime;
pub mod sysinfo;
pub mod tokenizer;
pub mod turboquant;
pub mod weights;
#[cfg(feature = "predictor")]
pub mod predictor;
#[cfg(feature = "server")]
pub mod metrics;
#[cfg(feature = "server")]
pub mod server;

// Re-exports
pub use gguf::GgufFile;
pub use model::{InferenceContext, ModelConfig};

/// Result type alias with anyhow::Error
pub type Result<T> = std::result::Result<T, anyhow::Error>;
