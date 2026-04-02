//! Mixture-of-Experts (MoE) inference
//!
//! Implements token-level expert routing for MoE models like Qwen3.5-35B-A3B.
//! Only the top-k experts are computed per token, reducing active compute to
//! ~3B params instead of 35B total.
//!
//! Architecture (per MoE layer):
//! 1. Router gate: `x @ gate_inp_weight` → expert logits [n_experts]
//! 2. Top-k selection: pick k experts with highest logits
//! 3. Routing weights: softmax over selected expert logits
//! 4. For each selected expert: SwiGLU FFN
//! 5. Weighted sum: combine expert outputs by routing weights
//!
//! GGUF tensor naming for MoE layers:
//! - `blk.{i}.ffn_gate_inp.weight` — router gate [n_embd, n_experts]
//! - `blk.{i}.ffn_gate_exps.weight` — expert gate weights (packed)
//! - `blk.{i}.ffn_up_exps.weight` — expert up weights (packed)
//! - `blk.{i}.ffn_down_exps.weight` — expert down weights (packed)

use crate::ops;

/// Routing result for a single token
#[derive(Debug, Clone)]
pub struct RoutingResult {
    /// Indices of selected experts [k]
    pub expert_indices: Vec<usize>,
    /// Routing weights for selected experts [k] (softmax-normalized)
    pub expert_weights: Vec<f32>,
}

/// MoE router: selects top-k experts for each token
///
/// Takes input activations, projects through the gate matrix to get
/// per-expert logits, then selects top-k with softmax weighting.
pub struct MoeRouter {
    /// Gate weight matrix [n_embd, n_experts] (f32, row-major)
    gate_weight: Vec<f32>,
    /// Number of input dimensions
    n_embd: usize,
    /// Number of experts
    n_experts: usize,
    /// Number of experts to route to per token
    top_k: usize,
}

impl MoeRouter {
    /// Create a new MoE router.
    ///
    /// - `gate_weight`: router gate matrix [n_embd * n_experts], row-major
    /// - `n_embd`: embedding dimension
    /// - `n_experts`: total number of experts
    /// - `top_k`: number of experts to select per token
    pub fn new(gate_weight: Vec<f32>, n_embd: usize, n_experts: usize, top_k: usize) -> Self {
        assert_eq!(gate_weight.len(), n_embd * n_experts);
        assert!(top_k <= n_experts, "top_k must be <= n_experts");
        assert!(top_k >= 1, "top_k must be >= 1");

        Self {
            gate_weight,
            n_embd,
            n_experts,
            top_k,
        }
    }

    /// Route a single token to its top-k experts.
    ///
    /// Returns expert indices and softmax-normalized routing weights.
    pub fn route(&self, x: &[f32]) -> RoutingResult {
        debug_assert_eq!(x.len(), self.n_embd);

        // Compute expert logits: logits[j] = sum_i(x[i] * gate[i * n_experts + j])
        // Gate weight is [n_embd, n_experts] row-major, so we do transposed matvec
        let mut logits = vec![0.0f32; self.n_experts];
        ops::matvec_t(
            &mut logits,
            x,
            &self.gate_weight,
            self.n_embd,
            self.n_experts,
        );

        // Find top-k by partial sort
        let mut indices: Vec<usize> = (0..self.n_experts).collect();
        indices.select_nth_unstable_by(self.top_k - 1, |&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(self.top_k);

        // Sort selected indices for deterministic ordering
        indices.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Softmax over selected logits
        let selected_logits: Vec<f32> = indices.iter().map(|&i| logits[i]).collect();
        let mut weights = selected_logits.clone();
        ops::softmax(&mut weights);

        RoutingResult {
            expert_indices: indices,
            expert_weights: weights,
        }
    }

    /// Number of experts
    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    /// Top-k value
    pub fn top_k(&self) -> usize {
        self.top_k
    }
}

/// Compute a single expert's SwiGLU FFN output.
///
/// This reuses the existing `ops::ffn_swiglu` function but operates
/// on per-expert weights that may have been lazily dequantized.
///
/// - `x`: input activations [n_embd]
/// - `gate_w`: expert gate weight [n_embd * expert_ffn_dim]
/// - `up_w`: expert up weight [n_embd * expert_ffn_dim]
/// - `down_w`: expert down weight [expert_ffn_dim * n_embd]
/// - `n_embd`: embedding dimension
/// - `expert_ffn_dim`: expert intermediate dimension
///
/// Returns expert output [n_embd]
pub fn expert_forward(
    x: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    n_embd: usize,
    expert_ffn_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_embd];
    ops::ffn_swiglu(&mut out, x, gate_w, up_w, down_w, n_embd, expert_ffn_dim);
    out
}

/// Trait for providing expert weights (either pre-loaded or lazy).
pub trait ExpertProvider {
    /// Get dequantized weights for expert `idx`: (gate, up, down).
    ///
    /// For `ExpertWeights`, this returns references (zero-copy).
    /// For `LazyExpertWeights`, this dequantizes on demand.
    fn get_expert_f32(&self, idx: usize) -> anyhow::Result<(Vec<f32>, Vec<f32>, Vec<f32>)>;

    /// Number of experts
    fn n_experts(&self) -> usize;
}

impl ExpertProvider for ExpertWeights {
    fn get_expert_f32(&self, idx: usize) -> anyhow::Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (g, u, d) = self.get_expert(idx);
        Ok((g.to_vec(), u.to_vec(), d.to_vec()))
    }

    fn n_experts(&self) -> usize {
        self.n_experts()
    }
}

impl ExpertProvider for LazyExpertWeights {
    fn get_expert_f32(&self, idx: usize) -> anyhow::Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        self.get_expert(idx)
    }

    fn n_experts(&self) -> usize {
        self.n_experts()
    }
}

/// Run the full MoE forward pass for a single token.
///
/// 1. Route token to top-k experts
/// 2. Compute each expert's SwiGLU output
/// 3. Weighted sum of expert outputs
///
/// - `x`: input activations [n_embd]
/// - `router`: the MoE router
/// - `expert_provider`: provides expert weights (ExpertWeights or LazyExpertWeights)
/// - `n_embd`: embedding dimension
/// - `expert_ffn_dim`: expert intermediate dimension
///
/// Returns output [n_embd]
pub fn moe_forward(
    x: &[f32],
    router: &MoeRouter,
    expert_provider: &dyn ExpertProvider,
    n_embd: usize,
    expert_ffn_dim: usize,
) -> Vec<f32> {
    let routing = router.route(x);

    let mut out = vec![0.0f32; n_embd];

    for (i, &expert_idx) in routing.expert_indices.iter().enumerate() {
        let weight = routing.expert_weights[i];

        let (gate_w, up_w, down_w) = expert_provider
            .get_expert_f32(expert_idx)
            .expect("Failed to load expert weights");

        let expert_out = expert_forward(x, &gate_w, &up_w, &down_w, n_embd, expert_ffn_dim);

        for (o, &e) in out.iter_mut().zip(expert_out.iter()) {
            *o += weight * e;
        }
    }

    out
}

/// Container for expert weights (all experts in f32).
///
/// For models where all experts are loaded into memory (e.g., small MoE models
/// or when enough RAM is available). For large models, use `LazyExpertWeights`
/// which dequantizes experts on demand.
pub struct ExpertWeights {
    /// Gate weights per expert: expert_gate[i] = [n_embd * expert_ffn_dim]
    expert_gate: Vec<Vec<f32>>,
    /// Up weights per expert
    expert_up: Vec<Vec<f32>>,
    /// Down weights per expert
    expert_down: Vec<Vec<f32>>,
}

impl ExpertWeights {
    /// Create from pre-loaded expert weight vectors.
    pub fn new(
        expert_gate: Vec<Vec<f32>>,
        expert_up: Vec<Vec<f32>>,
        expert_down: Vec<Vec<f32>>,
    ) -> Self {
        let n = expert_gate.len();
        assert_eq!(expert_up.len(), n);
        assert_eq!(expert_down.len(), n);

        Self {
            expert_gate,
            expert_up,
            expert_down,
        }
    }

    /// Get weights for a specific expert: (gate, up, down)
    pub fn get_expert(&self, idx: usize) -> (&[f32], &[f32], &[f32]) {
        (
            &self.expert_gate[idx],
            &self.expert_up[idx],
            &self.expert_down[idx],
        )
    }

    /// Number of experts
    pub fn n_experts(&self) -> usize {
        self.expert_gate.len()
    }
}

/// Container for MoE layer configuration.
#[derive(Debug, Clone)]
pub struct MoeLayerConfig {
    /// Number of routed experts
    pub n_experts: usize,
    /// Number of experts selected per token
    pub top_k: usize,
    /// Expert FFN intermediate dimension
    pub expert_ffn_dim: usize,
    /// Embedding dimension
    pub n_embd: usize,
}

/// Lazy expert weight loader: dequantizes experts on demand.
///
/// Stores raw quantized bytes per expert (from GGUF Q4_K_M, Q8_0, etc.)
/// and only dequantizes when an expert is actually selected. This is critical
/// for large MoE models (35B+) where dequantizing all experts to f32 would
/// require ~280GB of RAM.
///
/// For the 35B-A3B model:
/// - Total expert weights (Q4_K_M): ~22GB across all layers
/// - Active experts per token: 8 out of 64
/// - Active dequantized weights: ~1.7GB (fits in RAM)
pub struct LazyExpertWeights {
    /// Raw quantized gate bytes per expert
    gate_raw: Vec<Vec<u8>>,
    /// Raw quantized up bytes per expert
    up_raw: Vec<Vec<u8>>,
    /// Raw quantized down bytes per expert
    down_raw: Vec<Vec<u8>>,
    /// Quantization type for all experts
    quant_type: crate::quant::QuantizationType,
    /// Number of experts
    n_experts: usize,
    /// Embedding dimension
    n_embd: usize,
    /// Expert FFN intermediate dimension
    expert_ffn_dim: usize,
    /// Cache of recently dequantized gate weights: (expert_idx, f32_data)
    gate_cache: std::cell::RefCell<Vec<Option<Vec<f32>>>>,
    /// Cache of recently dequantized up weights
    up_cache: std::cell::RefCell<Vec<Option<Vec<f32>>>>,
    /// Cache of recently dequantized down weights
    down_cache: std::cell::RefCell<Vec<Option<Vec<f32>>>>,
}

impl LazyExpertWeights {
    /// Create from raw quantized bytes.
    ///
    /// - `gate_raw`: quantized gate weight bytes per expert
    /// - `up_raw`: quantized up weight bytes per expert
    /// - `down_raw`: quantized down weight bytes per expert
    /// - `quant_type`: quantization format (e.g., Q4_K_M)
    /// - `n_embd`: embedding dimension
    /// - `expert_ffn_dim`: expert intermediate dimension
    pub fn new(
        gate_raw: Vec<Vec<u8>>,
        up_raw: Vec<Vec<u8>>,
        down_raw: Vec<Vec<u8>>,
        quant_type: crate::quant::QuantizationType,
        n_embd: usize,
        expert_ffn_dim: usize,
    ) -> Self {
        let n_experts = gate_raw.len();
        assert_eq!(up_raw.len(), n_experts);
        assert_eq!(down_raw.len(), n_experts);

        Self {
            gate_raw,
            up_raw,
            down_raw,
            quant_type,
            n_experts,
            n_embd,
            expert_ffn_dim,
            gate_cache: std::cell::RefCell::new(vec![None; n_experts]),
            up_cache: std::cell::RefCell::new(vec![None; n_experts]),
            down_cache: std::cell::RefCell::new(vec![None; n_experts]),
        }
    }

    /// Get dequantized weights for a specific expert (gate, up, down).
    ///
    /// Dequantizes on first access and caches the result.
    /// Subsequent calls for the same expert return cached f32 data.
    pub fn get_expert(&self, idx: usize) -> anyhow::Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        use crate::quant;

        // Gate
        let mut gate_cache = self.gate_cache.borrow_mut();
        if gate_cache[idx].is_none() {
            let deq = quant::dequantize(
                &self.gate_raw[idx],
                self.quant_type,
                1,
                self.n_embd * self.expert_ffn_dim,
            )?;
            gate_cache[idx] = Some(deq);
        }
        let gate = gate_cache[idx].clone().unwrap();

        // Up
        let mut up_cache = self.up_cache.borrow_mut();
        if up_cache[idx].is_none() {
            let deq = quant::dequantize(
                &self.up_raw[idx],
                self.quant_type,
                1,
                self.n_embd * self.expert_ffn_dim,
            )?;
            up_cache[idx] = Some(deq);
        }
        let up = up_cache[idx].clone().unwrap();

        // Down
        let mut down_cache = self.down_cache.borrow_mut();
        if down_cache[idx].is_none() {
            let deq = quant::dequantize(
                &self.down_raw[idx],
                self.quant_type,
                1,
                self.expert_ffn_dim * self.n_embd,
            )?;
            down_cache[idx] = Some(deq);
        }
        let down = down_cache[idx].clone().unwrap();

        Ok((gate, up, down))
    }

    /// Clear the dequantization cache (free memory).
    pub fn clear_cache(&self) {
        self.gate_cache.borrow_mut().fill(None);
        self.up_cache.borrow_mut().fill(None);
        self.down_cache.borrow_mut().fill(None);
    }

    /// Number of cached experts (gate side)
    pub fn cached_count(&self) -> usize {
        self.gate_cache
            .borrow()
            .iter()
            .filter(|c| c.is_some())
            .count()
    }

    /// Number of experts
    pub fn n_experts(&self) -> usize {
        self.n_experts
    }
}

/// MoE GGUF loader: reads expert weights from a GGUF file for a single MoE layer.
///
/// Handles the packed tensor format where all experts' weights are concatenated
/// into a single tensor per weight type (gate/up/down). Extracts per-expert
/// raw bytes for lazy dequantization.
pub struct MoELayerLoader {
    /// Path to GGUF file
    path: std::path::PathBuf,
    /// Quantization type
    quant_type: crate::quant::QuantizationType,
    /// Number of experts
    n_experts: usize,
    /// Embedding dimension
    n_embd: usize,
    /// Expert FFN intermediate dimension
    expert_ffn_dim: usize,
}

impl MoELayerLoader {
    /// Create a loader for a single MoE layer from GGUF metadata.
    ///
    /// - `gguf`: the opened GGUF file
    /// - `layer_idx`: which transformer layer (0-indexed)
    /// - `config`: MoE configuration from the GGUF metadata
    pub fn new(
        gguf: &crate::gguf::GgufFile,
        layer_idx: usize,
        config: &crate::gguf::MoeConfig,
    ) -> anyhow::Result<Self> {
        // Verify that MoE tensors exist for this layer
        let gate_name = format!("blk.{layer_idx}.ffn_gate_exps.weight");
        let tensor = gguf
            .tensors()
            .iter()
            .find(|t| t.name == gate_name)
            .ok_or_else(|| anyhow::anyhow!("MoE tensor not found: {gate_name}"))?;

        let gt: gguf_rs::GGMLType = tensor
            .kind
            .try_into()
            .map_err(|_| anyhow::anyhow!("Failed to parse GGML type"))?;
        let quant_type = crate::quant::QuantizationType::from_ggml_type(gt)?;

        Ok(Self {
            path: gguf.path().to_path_buf(),
            quant_type,
            n_experts: config.expert_count,
            n_embd: tensor.shape[0] as usize / config.expert_count,
            expert_ffn_dim: config.expert_intermediate_size,
        })
    }

    /// Load expert weights from GGUF for this MoE layer.
    ///
    /// Reads raw quantized bytes per expert (NOT dequantized). Dequantization
    /// happens lazily when `LazyExpertWeights::get_expert()` is called.
    ///
    /// Also loads the router gate weights (dequantized to f32 since it's small).
    pub fn load(&self, gguf: &crate::gguf::GgufFile) -> anyhow::Result<MoELayerWeights> {
        let layer_idx = self.find_layer_idx(gguf)?;

        // Load router gate (small, dequantize immediately)
        let router_gate =
            self.load_and_dequant_tensor(gguf, &format!("blk.{layer_idx}.ffn_gate_inp.weight"))?;

        // Load expert weights (raw bytes, lazy dequant)
        let gate_raw =
            self.load_expert_raw(gguf, &format!("blk.{layer_idx}.ffn_gate_exps.weight"))?;
        let up_raw = self.load_expert_raw(gguf, &format!("blk.{layer_idx}.ffn_up_exps.weight"))?;
        let down_raw =
            self.load_expert_raw(gguf, &format!("blk.{layer_idx}.ffn_down_exps.weight"))?;

        let lazy = LazyExpertWeights::new(
            gate_raw,
            up_raw,
            down_raw,
            self.quant_type,
            self.n_embd,
            self.expert_ffn_dim,
        );

        Ok(MoELayerWeights {
            router_gate,
            expert_weights: lazy,
        })
    }

    fn find_layer_idx(&self, gguf: &crate::gguf::GgufFile) -> anyhow::Result<usize> {
        for t in gguf.tensors() {
            if t.name.starts_with("blk.") && t.name.contains("ffn_gate_inp") {
                let parts: Vec<&str> = t.name.split('.').collect();
                if parts.len() >= 2 {
                    return Ok(parts[1].parse()?);
                }
            }
        }
        Err(anyhow::anyhow!("No MoE router gate tensor found in GGUF"))
    }

    fn load_and_dequant_tensor(
        &self,
        gguf: &crate::gguf::GgufFile,
        name: &str,
    ) -> anyhow::Result<Vec<f32>> {
        use std::io::{BufReader, Read, Seek};

        let tensor = gguf
            .tensors()
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found: {name}"))?;

        let total: usize = tensor.shape.iter().map(|&d| d as usize).product();

        let f = std::fs::File::open(&self.path)?;
        let mut r = BufReader::new(f);
        let ds = self.header_end(gguf)?;
        r.seek(std::io::SeekFrom::Start(ds + tensor.offset))?;
        let byte_size = crate::weights::tensor_byte_size(self.quant_type, total);
        let mut buf = vec![0u8; byte_size];
        r.read_exact(&mut buf)?;
        crate::quant::dequantize(&buf, self.quant_type, 1, total)
    }

    fn load_expert_raw(
        &self,
        gguf: &crate::gguf::GgufFile,
        name: &str,
    ) -> anyhow::Result<Vec<Vec<u8>>> {
        let tensor = gguf
            .tensors()
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found: {name}"))?;

        // Shape for MoE expert tensor: [n_experts * n_rows, n_cols]
        // where n_rows = n_embd for gate/up, expert_ffn_dim for down
        let total_rows = tensor.shape[0] as usize;
        let n_cols = tensor.shape[1] as usize;
        let rows_per_expert = total_rows / self.n_experts;

        // Compute bytes per expert
        let elements_per_expert = rows_per_expert * n_cols;
        let bytes_per_expert =
            crate::weights::tensor_byte_size(self.quant_type, elements_per_expert);

        // Read the full tensor raw bytes
        let ds = self.header_end(gguf)?;
        let full_bytes = crate::weights::read_raw_bytes(
            &self.path,
            ds + tensor.offset,
            self.n_experts * bytes_per_expert,
        )?;

        // Split into per-expert slices
        let mut expert_raw = Vec::with_capacity(self.n_experts);
        for e in 0..self.n_experts {
            let start = e * bytes_per_expert;
            let end = start + bytes_per_expert;
            expert_raw.push(full_bytes[start..end].to_vec());
        }

        Ok(expert_raw)
    }

    fn header_end(&self, _gguf: &crate::gguf::GgufFile) -> anyhow::Result<u64> {
        // Parse the GGUF header to find where tensor data starts
        use std::io::{Read, Seek};
        let mut f = std::fs::File::open(&self.path)?;
        let mut b8 = [0u8; 8];
        f.seek(std::io::SeekFrom::Start(8))?;
        f.read_exact(&mut b8)?;
        let nt = u64::from_le_bytes(b8);
        f.read_exact(&mut b8)?;
        let nm = u64::from_le_bytes(b8);
        let mut b4 = [0u8; 4];
        for _ in 0..nm {
            f.read_exact(&mut b8)?;
            f.seek(std::io::SeekFrom::Current(
                i64::try_from(u64::from_le_bytes(b8)).unwrap_or(0),
            ))?;
            f.read_exact(&mut b4)?;
            let vt = u32::from_le_bytes(b4);
            Self::skip_val(&mut f, vt)?;
        }
        for _ in 0..nt {
            f.read_exact(&mut b8)?;
            f.seek(std::io::SeekFrom::Current(
                i64::try_from(u64::from_le_bytes(b8)).unwrap_or(0),
            ))?;
            f.read_exact(&mut b4)?;
            let nd = u32::from_le_bytes(b4);
            f.seek(std::io::SeekFrom::Current(i64::from(nd) * 8 + 12))?;
        }
        Ok((f.stream_position()? + 31) & !31)
    }

    fn skip_val(f: &mut std::fs::File, vt: u32) -> anyhow::Result<()> {
        use std::io::{Read, Seek};
        let mut b8 = [0u8; 8];
        match vt {
            0..=1 | 7 => {
                f.seek(std::io::SeekFrom::Current(1))?;
            }
            2..=3 => {
                f.seek(std::io::SeekFrom::Current(2))?;
            }
            4..=6 => {
                f.seek(std::io::SeekFrom::Current(4))?;
            }
            8 => {
                f.read_exact(&mut b8)?;
                f.seek(std::io::SeekFrom::Current(
                    i64::try_from(u64::from_le_bytes(b8)).unwrap_or(0),
                ))?;
            }
            9 => {
                let mut b4 = [0u8; 4];
                f.read_exact(&mut b4)?;
                let et = u32::from_le_bytes(b4);
                f.read_exact(&mut b8)?;
                let n = u64::from_le_bytes(b8);
                if et == 8 {
                    for _ in 0..n {
                        f.read_exact(&mut b8)?;
                        f.seek(std::io::SeekFrom::Current(
                            i64::try_from(u64::from_le_bytes(b8)).unwrap_or(0),
                        ))?;
                    }
                } else {
                    let es: u64 = match et {
                        0..=1 | 7 => 1,
                        2..=3 => 2,
                        4..=6 => 4,
                        10..=12 => 8,
                        _ => 0,
                    };
                    f.seek(std::io::SeekFrom::Current(
                        i64::try_from(n * es).unwrap_or(0),
                    ))?;
                }
            }
            10..=12 => {
                f.seek(std::io::SeekFrom::Current(8))?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Number of experts
    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    /// Embedding dimension
    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    /// Expert FFN intermediate dimension
    pub fn expert_ffn_dim(&self) -> usize {
        self.expert_ffn_dim
    }

    /// Quantization type
    pub fn quant_type(&self) -> crate::quant::QuantizationType {
        self.quant_type
    }
}

/// Loaded weights for a single MoE layer.
pub struct MoELayerWeights {
    /// Router gate weights [n_embd * n_experts] (f32, row-major)
    pub router_gate: Vec<f32>,
    /// Expert weights (lazy dequantization)
    pub expert_weights: LazyExpertWeights,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_selects_top_k() {
        let n_embd = 4;
        let n_experts = 8;
        let top_k = 2;

        // Gate weight: identity-like (each dim maps to different expert)
        let mut gate_weight = vec![0.0f32; n_embd * n_experts];
        // dim 0 → expert 0, dim 1 → expert 1, etc.
        for i in 0..n_embd.min(n_experts) {
            gate_weight[i * n_experts + i] = 1.0;
        }

        let router = MoeRouter::new(gate_weight, n_embd, n_experts, top_k);

        // Input: [1, 0, 0, 0] → should select expert 0
        let x = [1.0, 0.0, 0.0, 0.0f32];
        let result = router.route(&x);
        assert_eq!(result.expert_indices.len(), top_k);
        assert_eq!(result.expert_indices[0], 0);

        // Weights should sum to 1 (softmax)
        let weight_sum: f32 = result.expert_weights.iter().sum();
        assert!(
            (weight_sum - 1.0).abs() < 1e-5,
            "Weights should sum to 1, got {weight_sum}"
        );
    }

    #[test]
    fn test_router_different_inputs() {
        let n_embd = 4;
        let n_experts = 4;
        let top_k = 2;

        // Each dim maps to one expert with weight 1.0
        let mut gate_weight = vec![0.0f32; n_embd * n_experts];
        for i in 0..n_embd {
            gate_weight[i * n_experts + i] = 1.0;
        }

        let router = MoeRouter::new(gate_weight, n_embd, n_experts, top_k);

        // Input: [0, 0, 1, 0] → should select expert 2
        let x = [0.0, 0.0, 1.0, 0.0f32];
        let result = router.route(&x);
        assert_eq!(result.expert_indices[0], 2);
    }

    #[test]
    fn test_expert_swiglu() {
        let n_embd = 4;
        let expert_ffn_dim = 8;

        // Simple identity-like weights for testing
        let gate_w = vec![1.0; n_embd * expert_ffn_dim];
        let up_w = vec![0.5; n_embd * expert_ffn_dim];
        let down_w = vec![0.1; expert_ffn_dim * n_embd];

        let x = [1.0, 2.0, 3.0, 4.0f32];
        let out = expert_forward(&x, &gate_w, &up_w, &down_w, n_embd, expert_ffn_dim);

        assert_eq!(out.len(), n_embd);
        // All outputs should be finite
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_moe_forward() {
        let n_embd = 4;
        let n_experts = 4;
        let top_k = 2;
        let expert_ffn_dim = 8;

        // Router: each dim maps to one expert
        let mut gate_weight = vec![0.0f32; n_embd * n_experts];
        for i in 0..n_embd {
            gate_weight[i * n_experts + i] = 1.0;
        }
        let router = MoeRouter::new(gate_weight, n_embd, n_experts, top_k);

        // Expert weights: all experts have simple weights
        let expert_gate: Vec<Vec<f32>> = (0..n_experts)
            .map(|_| vec![0.1; n_embd * expert_ffn_dim])
            .collect();
        let expert_up: Vec<Vec<f32>> = (0..n_experts)
            .map(|_| vec![0.2; n_embd * expert_ffn_dim])
            .collect();
        let expert_down: Vec<Vec<f32>> = (0..n_experts)
            .map(|_| vec![0.05; expert_ffn_dim * n_embd])
            .collect();
        let weights = ExpertWeights::new(expert_gate, expert_up, expert_down);

        let x = [1.0, 0.0, 0.0, 0.0f32];
        let out = moe_forward(&x, &router, &weights, n_embd, expert_ffn_dim);

        assert_eq!(out.len(), n_embd);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_routing_weights_are_normalized() {
        let n_embd = 8;
        let n_experts = 16;
        let top_k = 4;

        // Random-ish gate weights
        let gate_weight: Vec<f32> = (0..n_embd * n_experts)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();

        let router = MoeRouter::new(gate_weight, n_embd, n_experts, top_k);

        let x: Vec<f32> = (0..n_embd).map(|i| (i as f32 * 0.3).cos()).collect();
        let result = router.route(&x);

        assert_eq!(result.expert_indices.len(), top_k);
        assert_eq!(result.expert_weights.len(), top_k);

        // Weights should sum to ~1
        let sum: f32 = result.expert_weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Routing weights should sum to 1, got {sum}"
        );

        // All weights should be positive
        assert!(result.expert_weights.iter().all(|&w| w > 0.0));

        // All indices should be distinct
        let mut sorted = result.expert_indices.clone();
        sorted.sort();
        for i in 1..sorted.len() {
            assert_ne!(
                sorted[i],
                sorted[i - 1],
                "Expert indices should be distinct"
            );
        }
    }

    #[test]
    fn test_lazy_expert_weights() {
        let n_embd = 4;
        let n_experts = 4;
        let expert_ffn_dim = 8;

        // Create fake F32 expert weight bytes
        let gate_raw: Vec<Vec<u8>> = (0..n_experts)
            .map(|e| {
                let weights: Vec<f32> = (0..n_embd * expert_ffn_dim)
                    .map(|i| (e * 100 + i) as f32 * 0.01)
                    .collect();
                weights.iter().flat_map(|f| f.to_le_bytes()).collect()
            })
            .collect();
        let up_raw: Vec<Vec<u8>> = (0..n_experts)
            .map(|e| {
                let weights: Vec<f32> = (0..n_embd * expert_ffn_dim)
                    .map(|i| (e * 100 + i) as f32 * 0.02)
                    .collect();
                weights.iter().flat_map(|f| f.to_le_bytes()).collect()
            })
            .collect();
        let down_raw: Vec<Vec<u8>> = (0..n_experts)
            .map(|e| {
                let weights: Vec<f32> = (0..expert_ffn_dim * n_embd)
                    .map(|i| (e * 100 + i) as f32 * 0.005)
                    .collect();
                weights.iter().flat_map(|f| f.to_le_bytes()).collect()
            })
            .collect();

        let lazy = LazyExpertWeights::new(
            gate_raw,
            up_raw,
            down_raw,
            crate::quant::QuantizationType::F32,
            n_embd,
            expert_ffn_dim,
        );

        // Initially no experts are cached
        assert_eq!(lazy.cached_count(), 0);

        // Get expert 0 — should dequantize and cache
        let (gate, up, down) = lazy.get_expert(0).unwrap();
        assert_eq!(gate.len(), n_embd * expert_ffn_dim);
        assert_eq!(up.len(), n_embd * expert_ffn_dim);
        assert_eq!(down.len(), expert_ffn_dim * n_embd);
        assert_eq!(lazy.cached_count(), 1);

        // Get expert 1 — should increment cache
        let _ = lazy.get_expert(1).unwrap();
        assert_eq!(lazy.cached_count(), 2);

        // Clear cache
        lazy.clear_cache();
        assert_eq!(lazy.cached_count(), 0);
    }

    #[test]
    fn test_moe_forward_with_lazy_weights() {
        let n_embd = 4;
        let n_experts = 4;
        let top_k = 2;
        let expert_ffn_dim = 8;

        // Router
        let mut gate_weight = vec![0.0f32; n_embd * n_experts];
        for i in 0..n_embd {
            gate_weight[i * n_experts + i] = 1.0;
        }
        let router = MoeRouter::new(gate_weight, n_embd, n_experts, top_k);

        // Lazy expert weights (F32)
        let gate_raw: Vec<Vec<u8>> = (0..n_experts)
            .map(|_| {
                vec![0.1f32; n_embd * expert_ffn_dim]
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect()
            })
            .collect();
        let up_raw: Vec<Vec<u8>> = (0..n_experts)
            .map(|_| {
                vec![0.2f32; n_embd * expert_ffn_dim]
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect()
            })
            .collect();
        let down_raw: Vec<Vec<u8>> = (0..n_experts)
            .map(|_| {
                vec![0.05f32; expert_ffn_dim * n_embd]
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect()
            })
            .collect();

        let lazy = LazyExpertWeights::new(
            gate_raw,
            up_raw,
            down_raw,
            crate::quant::QuantizationType::F32,
            n_embd,
            expert_ffn_dim,
        );

        let x = [1.0, 0.0, 0.0, 0.0f32];
        let out = moe_forward(&x, &router, &lazy, n_embd, expert_ffn_dim);

        assert_eq!(out.len(), n_embd);
        assert!(out.iter().all(|v| v.is_finite()));

        // Only top_k experts should have been cached
        assert_eq!(lazy.cached_count(), top_k);
    }
}
