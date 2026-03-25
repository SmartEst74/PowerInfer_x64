//! Vulkan/SPIR-V kernel modules
//!
//! Same kernels as CUDA but compiled for Vulkan compute shaders.
//! Uses rust-gpu's SPIR-V backend.

pub mod matmul;
pub mod fused_ops;
pub mod moe;
