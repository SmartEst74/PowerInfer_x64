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
//! let mut ctx = InferenceContext::from_gguf("/path/to/model.gguf", BackendFactory::cpu())?;
//!
//! let output = ctx.generate("Hello, world!", 100)?;
//! println!("{}", output);
//! # Ok(())
//! # }
//! ```

pub mod activation;
pub mod benchmark;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod gguf;
#[cfg(feature = "server")]
pub mod metrics;
pub mod model;
pub mod moe;
pub mod ops;
#[cfg(feature = "predictor")]
pub mod predictor;
pub mod quant;
pub mod runtime;
#[cfg(feature = "server")]
pub mod server;
pub mod simd;
pub mod ssm;
pub mod sysinfo;
pub mod tokenizer;
pub mod turboquant;
pub mod weights;

// Re-exports
pub use gguf::GgufFile;
pub use model::{GenerationOptions, InferenceContext, ModelConfig};

/// Result type alias with anyhow::Error
pub type Result<T> = std::result::Result<T, anyhow::Error>;
