//! CUDA kernel modules
//!
//! These kernels are written in Rust and compiled to PTX via rust-gpu.
//! At build time, they are compiled and embedded into the binary.

// Note: These are module declarations, actual kernel code in sub-files
// The build script compiles these .rs files as `#[no_std]` kernels.

pub mod matmul;
pub mod fused_ops;
pub mod moe;
