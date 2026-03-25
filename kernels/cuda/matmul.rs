//! CUDA kernel: sparse matrix multiplication with hot neuron selection
//!
//! This kernel computes: output[i] = sum_j (input[j] * weight[i, j])
//! but ONLY for neuron indices in `hot_indices` array.
//! Cold neurons are skipped entirely.
//!
//! # Safety
//! This is unsafe Rust that compiles to PTX via rust-gpu.
//! All pointer arguments must be valid device pointers.

#![no_std]
#![allow(non_snake_case)]
#![allow(unused_attributes)]

use rust_gpu::cuda::{kernel, thread, block, syncthreads};
use core::f16;

/// Sparse matmul kernel for hot neuron computation
///
/// # Arguments
/// - `hot_indices`: array of neuron output indices to compute (length = n_hot)
/// - `weights`: full weight matrix [n_out, n_in] in f16, row-major
/// - `input`: activation vector [n_in] in f16
/// - `output`: output vector [n_out] in f16; only hot entries are written
///
/// Each thread block computes one or more hot neurons.
#[kernel]
pub fn sparse_matmul_hot(
    hot_indices: *const u32,
    weights: *const f16,
    input: *const f16,
    output: *mut f16,
    n_out: usize,
    n_in: usize,
    n_hot: usize,
) {
    // Thread index determines which hot neuron to compute
    let tid = thread::idx();
    let block_size = block::dimx();
    
    // Each thread computes one neuron if within n_hot
    if tid < n_hot {
        let out_idx = unsafe { *hot_indices.add(tid) } as usize;
        if out_idx >= n_out {
            return;  // Invalid index, skip
        }
        
        // Pointer to row of weights for this neuron
        let row_ptr = unsafe { weights.add(out_idx * n_in) };
        
        // Compute dot product: sum_j input_j * weight[row, j]
        let mut sum = 0.0f32;  // accumulate in f32 for precision
        
        for j in 0..n_in {
            let w = unsafe { *row_ptr.add(j) }.to_f32();
            let x = unsafe { *input.add(j) }.to_f32();
            sum += w * x;
        }
        
        // Write output (convert back to f16)
        unsafe { *output.add(out_idx) = f16::from_f32(sum) };
    }
}

/// Fused kernel: RMSNorm + RoPE + SwiGLU + MatMul
///
/// This combines multiple operations to reduce memory traffic:
/// 1. Load input, apply RMSNorm
/// 2. Apply RoPE to positional embedding
/// 3. SwiGLU: gate * silu(x)
/// 4. MatMul with weights
///
/// All in one pass over the weight matrix.
#[kernel]
pub fn fused_rmsnorm_rope_swiglu_matmul(
    input: *const f16,
    weights: *const f16,
    rms_weights: *const f16,  // RMS norm learned weights
    rope_cos: *const f16,     // RoPE cos table
    rope_sin: *const f16,     // RoPE sin table
    output: *mut f16,
    seq_len: usize,
    hidden_dim: usize,
    head_dim: usize,
) {
    // Simplified: each thread computes one output element
    let tid = thread::idx();
    let total = seq_len * hidden_dim;
    
    if tid < total {
        let idx = tid;
        let s = idx / hidden_dim;  // sequence position
        let h = idx % hidden_dim;  // hidden dimension
        
        // 1. RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        // (In practice, precomputed RMS norms would be passed)
        let x = unsafe { *input.add(idx) }.to_f32();
        let rms_weight = unsafe { *rms_weights.add(h) }.to_f32();
        let x_norm = x * rms_weight;  // placeholder
        
        // 2. RoPE: apply rotation to head_dim dimensions
        let mut x_rope = x_norm;
        if h < head_dim {
            let rope_idx = s % 4096;  // RoPE table size
            let cos = unsafe { *rope_cos.add(rope_idx * (head_dim/2) + h/2) }.to_f32();
            let sin = unsafe { *rope_sin.add(rope_idx * (head_dim/2) + h/2) }.to_f32();
            if h % 2 == 0 {
                x_rope = x_norm * cos - (if h+1 < head_dim { unsafe { *input.add(idx+1) }.to_f32() } else { 0.0 }) * sin;
            } else {
                x_rope = (if h-1 >= 0 { unsafe { *input.add(idx-1) }.to_f32() } else { 0.0 }) * sin + x_norm * cos;
            }
        }
        
        // 3. SwiGLU: x * sigmoid(x)
        let x_swiglu = x_rope * (1.0 / (1.0 + (-x_rope).exp()));
        
        // 4. MatMul with weights would happen here
        // For demonstration, just write modified value
        unsafe { *output.add(idx) = f16::from_f32(x_swiglu) };
    }
}

/// MoE expert dispatch kernel
///
/// For each token in batch, route to top-k experts.
/// Reorganizes input tensor from [batch, seq, hidden] to [expert, tokens_for_expert, hidden].
#[kernel]
pub fn moe_dispatch(
    token_embeddings: *const f16,     // [batch * seq, hidden]
    expert_affinities: *const f32,    // [batch * seq, n_experts]
    output_buffers: *mut *mut f16,    // array of n_experts pointers, each [max_tokens_per_expert, hidden]
    token_to_expert: *mut u32,        // [batch * seq] maps token to selected expert
    batch_seq_len: usize,
    n_experts: usize,
    k: usize,                         // top-k
    hidden_dim: usize,
) {
    // Each thread handles one token's routing
    let tid = thread::idx();
    if tid < batch_seq_len {
        // Load affinities for this token
        let mut topk = [(0usize, f32::NEG_INFINITY); 8]; // max k=8
        for e in 0..n_experts {
            let score = unsafe { *expert_affinities.add(tid * n_experts + e) };
            // Insert into top-k
            for i in 0..k {
                if score > topk[i].1 {
                    // Shift lower scores down
                    for j in (i+1..k).rev() {
                        topk[j] = topk[j-1];
                    }
                    topk[i] = (e, score);
                    break;
                }
            }
        }
        
        // Dispatch to each expert
        for i in 0..k {
            let expert = topk[i].0;
            // In practice: atomic increment per-expert counter, then scatter
            // Simplified: just record mapping
            unsafe { *token_to_expert.add(tid) = expert as u32 };
        }
    }
}
