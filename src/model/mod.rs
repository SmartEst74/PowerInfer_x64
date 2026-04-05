//! Model architectures and inference logic
//!
//! Implements the Llama/Qwen-style transformer forward pass with:
//! - RMSNorm, RoPE, multi-head attention, SwiGLU FFN
//! - CPU inference with f32 weights

use std::{collections::BTreeMap, path::Path};

use crate::activation::{ActivationRecorder, HotNeuronIndex};
use crate::gguf::{GgufFile, SsmConfig};
use crate::ops;
use crate::quant::QuantizationType;
use crate::runtime::Backend;
use crate::sysinfo::{ExecutionPlan, HardwareProfile};
use crate::tokenizer::Tokenizer;
use crate::weights::Weights;
use crate::Result;

/// Per-layer attention configuration
#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub head_count: usize,
    pub head_count_kv: Option<usize>,
    /// Head dimension used for computing attention (key/query head dim)
    pub head_dim: usize,
    /// Explicit key length per head (may differ from head_dim for some architectures)
    pub key_length: Option<usize>,
    /// Explicit value length per head
    pub value_length: Option<usize>,
    /// RoPE rotation dimension count
    pub rope_dim: Option<usize>,
    pub full_attention_interval: Option<usize>,
}

impl LayerConfig {
    /// Head dimension for attention computation (key_length if set, else head_dim)
    pub fn kv_head_dim(&self) -> usize {
        self.key_length.unwrap_or(self.head_dim)
    }
    /// Value dimension per head
    pub fn v_head_dim(&self) -> usize {
        self.value_length.unwrap_or(self.head_dim)
    }
    /// Query dimension per head (may be larger than key_length for qwen35moe)
    pub fn q_head_dim(&self) -> usize {
        self.head_dim
    }
}

/// Complete model configuration
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arch: String,
    pub name: Option<String>,
    pub context_length: usize,
    pub embedding_length: usize,
    pub block_count: usize,
    pub attention: LayerConfig,
    pub feed_forward_length: usize,
    pub moe: Option<crate::gguf::MoeConfig>,
    pub ssm: Option<SsmConfig>,
    pub quantization: QuantizationType,
    pub rms_epsilon: f32,
    pub rope_freq_base: f32,
    /// RoPE dimension sections for IMRoPE (e.g., [11, 11, 10, 0] for Qwen3.5)
    pub rope_sections: Option<[i32; 4]>,
}

use crate::turboquant::CompressedKVCache;

/// KV cache for a single layer
/// Supports both f32 (default) and TurboQuant compressed (opt-in)
struct LayerCache {
    /// F32 key cache [seq_len * n_kv_heads * head_dim]
    key_cache: Vec<f32>,
    /// F32 value cache [seq_len * n_kv_heads * head_dim]
    value_cache: Vec<f32>,
    /// TurboQuant compressed KV cache (optional)
    compressed: Option<CompressedKVCache>,
}

impl LayerCache {
    fn new(compressed: bool, n_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            key_cache: Vec::new(),
            value_cache: Vec::new(),
            compressed: if compressed {
                Some(CompressedKVCache::new(n_kv_heads, head_dim))
            } else {
                None
            },
        }
    }

    fn seq_len(&self, head_dim: usize, n_heads: usize) -> usize {
        if let Some(ref cc) = self.compressed {
            return cc.seq_len();
        }
        let per_step = n_heads * head_dim;
        if per_step == 0 {
            0
        } else {
            self.key_cache.len() / per_step
        }
    }
}

struct CachedExpertWeights {
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

/// Inference context - main entry point for generation
pub struct InferenceContext {
    config: ModelConfig,
    backend: Box<dyn Backend>,
    weights: Weights,
    tokenizer: Tokenizer,
    layer_caches: Vec<LayerCache>,
    use_compressed_cache: bool,
    /// SSM hidden states per layer [layer_idx] = [d_state * d_inner]
    /// Only used for hybrid SSM+attention architectures (Qwen3.5)
    ssm_states: Vec<Vec<f32>>,
    /// SSM conv1d sliding window per layer: last (d_conv-1) input vectors.
    /// [layer_idx] = ring buffer of shape [d_conv-1][half_inner]
    ssm_conv_states: Vec<Vec<Vec<f32>>>,
    /// Hardware profile (filled on startup)
    hw_profile: Option<HardwareProfile>,
    /// Execution plan (filled on startup)
    exec_plan: Option<ExecutionPlan>,
    /// Optional activation recorder used by profiling runs.
    activation_recorder: Option<ActivationRecorder>,
    /// Optional per-layer hot-index plan used for sparse dense FFN execution.
    hot_index: Option<Vec<Option<Vec<usize>>>>,
    /// Lazily populated cache of hot MoE experts keyed by layer and expert id.
    hot_expert_cache: Vec<BTreeMap<usize, CachedExpertWeights>>,
    /// GPU resources: persistent CUDA contexts + pre-uploaded weight buffers.
    #[cfg(feature = "cuda")]
    gpu: Option<crate::cuda::cuda_impl::GpuResources>,
}

/// Generation controls for text decoding.
#[derive(Debug, Clone, Copy)]
pub struct GenerationOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl GenerationOptions {
    pub fn greedy(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            ..Self::default()
        }
    }
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            temperature: 0.0,
            top_p: 1.0,
            repetition_penalty: 1.1,
            seed: 0xD1CE_F00D_FEE1_DEAD,
        }
    }
}

impl InferenceContext {
    /// Build inference context from GGUF file
    pub fn from_gguf<P: AsRef<Path>>(gguf_path: P, backend: Box<dyn Backend>) -> Result<Self> {
        let gguf = GgufFile::open(&gguf_path)?;
        let config = gguf.model_config()?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        let weights = Weights::from_gguf(&gguf)?;

        let n_kv_heads = config
            .attention
            .head_count_kv
            .unwrap_or(config.attention.head_count);
        let kv_head_dim = config.attention.kv_head_dim();
        let block_count = config.block_count;
        // Use TurboQuant-compressed KV cache: 3-bit keys with QJL + f16 values.
        // Keys use TurboQuant (3-bit Lloyd-Max + 1-bit QJL = ~4 bits/coordinate) for
        // quality-neutral attention (paper: arXiv:2504.19874, 0.997 needle-in-haystack).
        // Values use f16 (2 bytes) for negligible precision loss on weighted sums.
        // DISABLED for debugging: testing if TurboQuant compression causes output degradation
        let use_compressed = false;

        let layer_caches = (0..config.block_count)
            .map(|layer_idx| {
                // Full attention layers may have different kv_head_dim than GDR layers.
                // Derive from attn_k tensor shape if available.
                let layer_kv_dim = weights
                    .get(&format!("blk.{layer_idx}.attn_k.weight"))
                    .and_then(|t| t.shape.get(1).copied())
                    .map(|total| total / n_kv_heads.max(1))
                    .unwrap_or(kv_head_dim);
                LayerCache::new(use_compressed, n_kv_heads, layer_kv_dim)
            })
            .collect();

        // Initialize SSM states for hybrid architectures (Qwen3.5 has SSM layers)
        // Use d_state and d_inner from config.ssm if available, else sensible defaults
        let (d_state, d_inner) = if let Some(ref ssm) = config.ssm {
            (ssm.state_size, ssm.inner_size)
        } else {
            (16, 2 * config.embedding_length)
        };
        let ssm_states: Vec<Vec<f32>> = (0..config.block_count)
            .map(|_| crate::ssm::create_ssm_state(d_state, d_inner))
            .collect();

        // Conv1d sliding window: d_conv-1 previous inputs per SSM layer.
        // Default d_conv = 4 → keep last 3 input vectors.
        let ssm_conv_states: Vec<Vec<Vec<f32>>> = (0..config.block_count)
            .map(|_| Vec::new()) // lazily populated on first use
            .collect();

        // --- Hardware sweep and execution plan ---
        let hw_profile = HardwareProfile::sweep();
        hw_profile.print_report();

        // Pre-warm the page cache for attention/SSM weights (the hot working set).
        // This starts async readahead so the first forward pass doesn't stall.
        #[cfg(unix)]
        {
            let mut prefetch_names = Vec::new();
            let suffixes = [
                "attn_qkv",
                "attn_gate",
                "ssm_out",
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                "attn_norm",
                "ffn_norm",
                "post_attention_norm",
                // MoE shared expert (always active — critical for page cache)
                "ffn_gate_shexp",
                "ffn_up_shexp",
                "ffn_down_shexp",
                // MoE router
                "ffn_gate_inp",
            ];
            for i in 0..config.block_count {
                for s in &suffixes {
                    prefetch_names.push(format!("blk.{i}.{s}.weight"));
                }
            }
            prefetch_names.push("token_embd.weight".into());
            prefetch_names.push("output.weight".into());
            prefetch_names.push("output_norm.weight".into());
            weights.prefetch(&prefetch_names);
            eprintln!(
                "[mmap] Prefetching {} weight regions into page cache",
                prefetch_names.len()
            );

            // mlock shared expert + router weights to prevent eviction.
            // These are accessed every token on every layer — ~131 MB total.
            let mut mlock_names = Vec::new();
            for i in 0..config.block_count {
                for s in &[
                    "ffn_gate_shexp",
                    "ffn_up_shexp",
                    "ffn_down_shexp",
                    "ffn_gate_inp",
                    "ffn_gate_inp_shexp",
                ] {
                    mlock_names.push(format!("blk.{i}.{s}.weight"));
                }
            }
            mlock_names.push("output_norm.weight".into());
            let locked = weights.mlock(&mlock_names);
            if locked > 0 {
                eprintln!(
                    "[mmap] Locked {:.1} MB of shared expert/router weights in RAM",
                    locked as f64 / 1e6
                );
            }
        }

        // Estimate bytes per layer from model file size and layer count.
        // Subtract embedding + output head (~2 × vocab × n_embd × quant_bytes).
        let file_bytes = std::fs::metadata(gguf_path.as_ref())
            .map(|m| m.len())
            .unwrap_or(0);
        let overhead_bytes = 2 * 248320 * config.embedding_length as u64; // embed + output
        let layer_bytes = if config.block_count > 0 {
            file_bytes.saturating_sub(overhead_bytes) / config.block_count as u64
        } else {
            0
        };

        let exec_plan = ExecutionPlan::build(
            &hw_profile,
            config.block_count,
            layer_bytes,
            config.moe.is_some(),
            config.embedding_length,
            n_kv_heads,
            config.attention.head_dim,
        );
        exec_plan.print_report();

        // --- GPU weight upload ---
        // Dequantize and upload attention/SSM projection weights for GPU-assigned layers.
        // The execution plan underestimates what fits because it uses total layer bytes
        // (including MoE experts). Actual GPU weights are only ~135 MB/layer (attention
        // projections), so we can fit all layers across available GPUs.
        #[cfg(feature = "cuda")]
        let gpu = {
            // Count available GPUs and their free VRAM from the hardware profile.
            let n_gpus = hw_profile.gpus.len();
            if n_gpus == 0 {
                None
            } else {
                // Estimate per-layer GPU weight size: ~135 MB f32 for SSM, ~109 MB for attention.
                // Conservative: use 140 MB per layer budget.
                let per_layer_budget: u64 = 140 * 1024 * 1024;
                let n_layers = config.block_count;

                // Compute how many layers fit on each GPU (leave 500 MB headroom for
                // temporary buffers, kernel stack, etc.)
                let headroom: u64 = 500 * 1024 * 1024;
                let gpu_caps: Vec<usize> = hw_profile
                    .gpus
                    .iter()
                    .map(|g| {
                        let usable = g.free_vram.saturating_sub(headroom);
                        (usable / per_layer_budget) as usize
                    })
                    .collect();

                // Total capacity across GPUs
                let total_cap: usize = gpu_caps.iter().sum();

                // Assign layers balanced across GPUs to minimize context switches
                // while keeping VRAM usage even (leaves room for LM head split).
                let layers_to_offload = n_layers.min(total_cap);
                let mut gpu_assignments: Vec<(usize, usize)> = Vec::new();
                let mut gpu_remaining = gpu_caps.clone();

                for layer_idx in 0..layers_to_offload {
                    // Find GPU with most remaining capacity
                    let best_gpu = gpu_remaining
                        .iter()
                        .enumerate()
                        .filter(|(_, &r)| r > 0)
                        .max_by_key(|(_, &r)| r)
                        .map(|(i, _)| i);
                    if let Some(gi) = best_gpu {
                        gpu_assignments.push((layer_idx, gi));
                        gpu_remaining[gi] -= 1;
                    }
                }

                eprintln!(
                    "[GPU] Offloading {}/{} layers ({} per GPU avg, {} GPUs)",
                    gpu_assignments.len(),
                    n_layers,
                    gpu_assignments.len() / n_gpus.max(1),
                    n_gpus,
                );

                if gpu_assignments.is_empty() {
                    None
                } else {
                    match crate::cuda::cuda_impl::GpuResources::init(
                        &gpu_assignments,
                        |layer_idx, suffix| {
                            let key = format!("blk.{layer_idx}.{suffix}.weight");
                            let tensor = weights.get(&key)?;
                            let data = tensor.to_f32().ok()?;
                            let n_in = tensor.shape.first().copied().unwrap_or(0);
                            let n_out = tensor.shape.get(1).copied().unwrap_or(0);
                            Some((data, n_out, n_in))
                        },
                    ) {
                        Ok(mut res) => {
                            // Upload LM head to GPU (split across GPUs)
                            let lm_tensor = weights
                                .get("output.weight")
                                .or_else(|| weights.get("token_embd.weight"));
                            if let Some(lm_t) = lm_tensor {
                                let n_in = lm_t.shape.first().copied().unwrap_or(0);
                                let n_out = lm_t.shape.get(1).copied().unwrap_or(0);
                                if let Ok(data) = lm_t.to_f32() {
                                    match res.upload_lm_head(&data, n_out, n_in) {
                                        Ok(()) => {}
                                        Err(e) => eprintln!("[GPU] LM head upload failed: {e}"),
                                    }
                                }
                            }
                            Some(res)
                        }
                        Err(e) => {
                            eprintln!("[GPU] Failed to initialize GPU resources: {e}");
                            None
                        }
                    }
                }
            }
        };

        Ok(Self {
            config,
            backend,
            weights,
            tokenizer,
            layer_caches,
            use_compressed_cache: use_compressed,
            ssm_states,
            ssm_conv_states,
            hw_profile: Some(hw_profile),
            exec_plan: Some(exec_plan),
            activation_recorder: None,
            hot_index: None,
            hot_expert_cache: (0..block_count).map(|_| BTreeMap::new()).collect(),
            #[cfg(feature = "cuda")]
            gpu,
        })
    }

    /// Generate text given a prompt
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let (output, _) =
            self.generate_timed_with_options(prompt, GenerationOptions::greedy(max_tokens))?;
        Ok(output)
    }

    /// Generate text with sampling options.
    pub fn generate_with_options(
        &mut self,
        prompt: &str,
        options: GenerationOptions,
    ) -> Result<String> {
        let (output, _) = self.generate_timed_with_options(prompt, options)?;
        Ok(output)
    }

    /// Generate text and return per-token timing (seconds per token).
    /// First entry is prefill time, remaining are decode times.
    pub fn generate_timed(
        &mut self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<(String, Vec<f64>)> {
        self.generate_timed_with_options(prompt, GenerationOptions::greedy(max_tokens))
    }

    /// Generate text and return per-token timing with sampling controls.
    pub fn generate_timed_with_options(
        &mut self,
        prompt: &str,
        options: GenerationOptions,
    ) -> Result<(String, Vec<f64>)> {
        self.reset();

        let input_ids = self.tokenizer.encode(prompt);

        if input_ids.is_empty() {
            return Ok((String::new(), Vec::new()));
        }

        let eos_id = self.tokenizer.eos_token_id();
        let mut generated = Vec::new();
        let mut token_times = Vec::new();
        let mut rng = SplitMix64::new(options.seed);

        // Prefill: process all prompt tokens
        let t0 = std::time::Instant::now();
        let mut logits = self.forward(&input_ids)?;
        token_times.push(t0.elapsed().as_secs_f64());

        // Debug: print logit stats to stderr
        if diagnostics_enabled() {
            let n_nan = logits.iter().filter(|x| x.is_nan()).count();
            let n_inf = logits.iter().filter(|x| x.is_infinite()).count();
            let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min_logit = logits.iter().copied().fold(f32::INFINITY, f32::min);
            let argmax = sample_argmax(&logits);
            eprintln!(
                "[DEBUG] logits: len={}, nan={n_nan}, inf={n_inf}, min={min_logit:.3}, max={max_logit:.3}, argmax={argmax}",
                logits.len()
            );
        }

        for _ in 0..options.max_tokens {
            // Apply repetition penalty to logits of already-generated tokens
            apply_repetition_penalty(&mut logits, &generated, options.repetition_penalty);
            let next_token = sample_token(&logits, options.temperature, options.top_p, &mut rng);

            // Check for EOS
            if Some(next_token) == eos_id {
                break;
            }

            generated.push(next_token);

            // Forward pass with just the new token
            let t0 = std::time::Instant::now();
            logits = self.forward(&[next_token])?;
            token_times.push(t0.elapsed().as_secs_f64());
        }

        let output = self.tokenizer.decode(&generated);
        Ok((output, token_times))
    }

    /// Generate text with streaming: calls `on_token` with each decoded text fragment
    /// as soon as it's produced. Returns the full output and per-token timings.
    pub fn generate_streaming<F>(
        &mut self,
        prompt: &str,
        options: GenerationOptions,
        mut on_token: F,
    ) -> Result<(String, Vec<f64>)>
    where
        F: FnMut(&str),
    {
        self.reset();

        let input_ids = self.tokenizer.encode(prompt);
        if input_ids.is_empty() {
            return Ok((String::new(), Vec::new()));
        }

        let eos_id = self.tokenizer.eos_token_id();
        let mut generated = Vec::new();
        let mut token_times = Vec::new();
        let mut rng = SplitMix64::new(options.seed);

        // Prefill
        let t0 = std::time::Instant::now();
        let mut logits = self.forward(&input_ids)?;
        token_times.push(t0.elapsed().as_secs_f64());

        for _ in 0..options.max_tokens {
            apply_repetition_penalty(&mut logits, &generated, options.repetition_penalty);
            let next_token = sample_token(&logits, options.temperature, options.top_p, &mut rng);

            if Some(next_token) == eos_id {
                break;
            }

            generated.push(next_token);

            // Decode and stream the new token immediately
            let text = self.tokenizer.decode(&[next_token]);
            on_token(&text);

            let t0 = std::time::Instant::now();
            logits = self.forward(&[next_token])?;
            token_times.push(t0.elapsed().as_secs_f64());
        }

        let output = self.tokenizer.decode(&generated);
        Ok((output, token_times))
    }

    /// Single forward pass through the model
    ///
    /// Returns logits over the vocabulary for the last position.
    /// This is public for benchmarking and profiling use.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let n_embd = self.config.embedding_length;
        let n_layers = self.config.block_count;
        let n_heads = self.config.attention.head_count;
        let n_kv_heads = self.config.attention.head_count_kv.unwrap_or(n_heads);
        // head_dim = q head dim (may be larger than kv_head_dim for some models)
        let q_head_dim = self.config.attention.q_head_dim();
        let kv_head_dim = self.config.attention.kv_head_dim();
        let v_head_dim = self.config.attention.v_head_dim();
        // RoPE: only rotate first rope_dim dims of each head (capped to kv_head_dim)
        let rope_dim = self
            .config
            .attention
            .rope_dim
            .unwrap_or(kv_head_dim)
            .min(kv_head_dim);
        let rms_eps = self.config.rms_epsilon;
        let rope_freq_base = self.config.rope_freq_base;

        // Pre-compute IMRoPE frequencies if model uses sectioned RoPE
        let imrope_freqs: Option<Vec<f32>> = self
            .config
            .rope_sections
            .as_ref()
            .map(|sections| ops::compute_imrope_freqs(rope_freq_base, rope_dim, sections));
        let n_tokens = tokens.len();
        let activation_recorder = self.activation_recorder.clone();
        // Physical core count drives parallelism decisions
        let cpu_cores = self
            .hw_profile
            .as_ref()
            .map_or(2, |hw| hw.cpu.physical_cores.max(1));

        // GPU dispatch: try GPU matvec first, fall back to CPU if not a GPU layer.
        // The macro returns true if GPU handled the matvec, false otherwise.
        #[cfg(feature = "cuda")]
        let gpu_ref = self.gpu.as_ref();

        /// Try GPU matvec for a layer's weight. Returns true if handled by GPU.
        macro_rules! try_gpu {
            ($out:expr, $inp:expr, $layer_idx:expr, $suffix:expr) => {{
                let mut _ok = false;
                #[cfg(feature = "cuda")]
                {
                    if let Some(gpu) = gpu_ref {
                        if let Some(r) = gpu.try_matvec($out, $inp, $layer_idx, $suffix) {
                            r.map_err(|e| anyhow::anyhow!("GPU matvec {}: {e}", $suffix))?;
                            _ok = true;
                        }
                    }
                }
                _ok
            }};
        }

        // --- Embedding lookup ---
        // token_embd.weight [ne0=n_embd, ne1=n_vocab] in GGML col-major
        // Token i's embedding is at flat positions [i*n_embd .. (i+1)*n_embd - 1]
        let embd_tensor = self
            .weights
            .get("token_embd.weight")
            .ok_or_else(|| anyhow::anyhow!("token_embd.weight not found"))?;
        let n_vocab = embd_tensor.shape.get(1).copied().unwrap_or(0);

        let mut x = vec![0.0f32; n_tokens * n_embd];
        for (t, &token_id) in tokens.iter().enumerate() {
            let id = token_id as usize;
            let emb = embd_tensor.embedding_row_to_f32(id)?;
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&emb);
        }

        // --- Pre-allocate scratch buffers (reused across all 40 layers) ---
        // Avoids hundreds of malloc/free calls per token in the hot decode loop.
        let mut scratch_layer_out = vec![0.0f32; n_tokens * n_embd];
        let mut scratch_normed = vec![0.0f32; n_embd];
        let mut scratch_ffn_out = vec![0.0f32; n_embd];

        // --- Layer loop ---
        for layer_idx in 0..n_layers {
            let layer_start = std::time::Instant::now();
            let prefix = format!("blk.{layer_idx}");

            // Detect layer type: SSM/GDR (has ssm_a) vs full-attention (has attn_q)
            let is_ssm = self.weights.has(&format!("{prefix}.ssm_a"));

            // Pre-attention RMSNorm weight
            let attn_norm_w = self
                .weights
                .get_data(&format!("{prefix}.attn_norm.weight"))?;

            let cache = &mut self.layer_caches[layer_idx];
            let prev_seq_len = cache.seq_len(kv_head_dim, n_kv_heads);

            scratch_layer_out.fill(0.0);

            for pos in 0..n_tokens {
                let pos_offset = pos * n_embd;
                let h = &x[pos_offset..pos_offset + n_embd];
                scratch_normed.fill(0.0);
                ops::rms_norm(&mut scratch_normed, h, &attn_norm_w, rms_eps);
                let normed = &scratch_normed[..];

                let attn_out = if is_ssm {
                    // --- Gated Delta Rule (linear attention) layer ---
                    // Reference: Qwen3_5GatedDeltaNet from HuggingFace transformers
                    //
                    // Tensors:
                    //   attn_qkv [n_embd, key_dim*2 + value_dim] = [2048, 8192]
                    //   attn_gate [n_embd, value_dim] = [2048, 4096]  (output gate Z)
                    //   ssm_conv1d [d_conv, conv_dim] = [4, 8192]  (depthwise conv)
                    //   ssm_alpha [n_embd, n_v_heads] = [2048, 32]  (decay input)
                    //   ssm_beta [n_embd, n_v_heads] = [2048, 32]  (write gate input)
                    //   ssm_a [n_v_heads] = [32]  (A_log: log decay parameter)
                    //   ssm_dt.bias [n_v_heads] = [32]  (dt bias)
                    //   ssm_norm [v_head_dim] = [128]  (RMSNormGated weight)
                    //   ssm_out [value_dim, n_embd] = [4096, 2048]  (output projection)

                    let inproj_t = self
                        .weights
                        .get(&format!("{prefix}.attn_qkv.weight"))
                        .ok_or_else(|| anyhow::anyhow!("{prefix}.attn_qkv.weight not found"))?;
                    let gate_t = self
                        .weights
                        .get(&format!("{prefix}.attn_gate.weight"))
                        .ok_or_else(|| anyhow::anyhow!("{prefix}.attn_gate.weight not found"))?;
                    let ssm_alpha_t = self
                        .weights
                        .get(&format!("{prefix}.ssm_alpha.weight"))
                        .ok_or_else(|| anyhow::anyhow!("{prefix}.ssm_alpha.weight not found"))?;
                    let ssm_beta_t = self
                        .weights
                        .get(&format!("{prefix}.ssm_beta.weight"))
                        .ok_or_else(|| anyhow::anyhow!("{prefix}.ssm_beta.weight not found"))?;
                    let ssm_out_t = self
                        .weights
                        .get(&format!("{prefix}.ssm_out.weight"))
                        .ok_or_else(|| anyhow::anyhow!("{prefix}.ssm_out.weight not found"))?;
                    let ssm_a_data = self.weights.get_data(&format!("{prefix}.ssm_a"))?;
                    let ssm_dt_bias = self.weights.get_data(&format!("{prefix}.ssm_dt.bias"))?;
                    let ssm_conv1d_w = self
                        .weights
                        .get_data(&format!("{prefix}.ssm_conv1d.weight"))?;
                    let ssm_norm_w = self
                        .weights
                        .get_data(&format!("{prefix}.ssm_norm.weight"))?;

                    // Derive dimensions from tensor shapes
                    let conv_dim = inproj_t.shape.get(1).copied().unwrap_or(0); // 8192
                    let value_dim = ssm_out_t.shape.first().copied().unwrap_or(0); // 4096
                    let key_dim = (conv_dim - value_dim) / 2; // 2048
                    let v_hd = ssm_norm_w.len(); // v_head_dim = 128
                    let n_v_h = if v_hd > 0 { value_dim / v_hd } else { 1 }; // 32
                    let k_hd = if n_v_h > 0 {
                        key_dim / (n_v_h / 2).max(1)
                    } else {
                        1
                    }; // 128
                    let n_k_h = if k_hd > 0 { key_dim / k_hd } else { 1 }; // 16
                    let d_conv = if conv_dim > 0 {
                        ssm_conv1d_w.len() / conv_dim
                    } else {
                        0
                    };

                    // Step 1: QKV projection
                    let mut qkv = vec![0.0f32; conv_dim];
                    if !try_gpu!(&mut qkv, normed, layer_idx, "attn_qkv") {
                        crate::quant::matvec_col_major(
                            &mut qkv,
                            normed,
                            inproj_t.raw(),
                            inproj_t.qtype,
                            n_embd,
                            conv_dim,
                        )?;
                    }

                    // Step 2: Depthwise causal conv1d over QKV channels + SiLU
                    if d_conv > 0 {
                        let conv_state = &mut self.ssm_conv_states[layer_idx];
                        if conv_state.is_empty() {
                            for _ in 0..d_conv.saturating_sub(1) {
                                conv_state.push(vec![0.0f32; conv_dim]);
                            }
                        }
                        conv_state.push(qkv.clone());
                        while conv_state.len() > d_conv {
                            conv_state.remove(0);
                        }
                        let n_hist = conv_state.len();
                        for i in 0..conv_dim {
                            let mut val = 0.0f32;
                            for (k, cs) in conv_state.iter().enumerate().take(n_hist) {
                                let kernel_idx = d_conv - n_hist + k;
                                // GGML layout [ne0=d_conv, ne1=conv_dim]: element [k, i] = data[i * d_conv + k]
                                let w_idx = i * d_conv + kernel_idx;
                                if w_idx < ssm_conv1d_w.len() {
                                    val += ssm_conv1d_w[w_idx] * cs[i];
                                }
                            }
                            // SiLU activation: x * sigmoid(x)
                            qkv[i] = val / (1.0 + (-val).exp());
                        }
                    }

                    // Step 3: Split into Q, K, V
                    let q_raw = &qkv[..key_dim]; // [2048] = 16 * 128
                    let k_raw = &qkv[key_dim..key_dim * 2]; // [2048] = 16 * 128
                    let v_raw = &qkv[key_dim * 2..]; // [4096] = 32 * 128

                    // Step 4: Z = output gate (from separate projection, NOT from QKV)
                    let mut z = vec![0.0f32; value_dim];
                    if !try_gpu!(&mut z, normed, layer_idx, "attn_gate") {
                        crate::quant::matvec_col_major(
                            &mut z,
                            normed,
                            gate_t.raw(),
                            gate_t.qtype,
                            n_embd,
                            value_dim,
                        )?;
                    }

                    // Step 5: Compute beta (write gate) and g (decay)
                    let mut beta_raw = vec![0.0f32; n_v_h];
                    crate::quant::matvec_col_major(
                        &mut beta_raw,
                        normed,
                        ssm_beta_t.raw(),
                        ssm_beta_t.qtype,
                        n_embd,
                        n_v_h,
                    )?;
                    // beta = sigmoid(beta_raw)
                    for b in beta_raw.iter_mut() {
                        *b = 1.0 / (1.0 + (-*b).exp());
                    }

                    let mut a_raw = vec![0.0f32; n_v_h];
                    crate::quant::matvec_col_major(
                        &mut a_raw,
                        normed,
                        ssm_alpha_t.raw(),
                        ssm_alpha_t.qtype,
                        n_embd,
                        n_v_h,
                    )?;
                    // g = -exp(A_log) * softplus(a + dt_bias)
                    // All SSM parameters are in TILED V head order in the GGUF:
                    // convert_hf_to_gguf.py reorders A_log, dt_bias, alpha, beta,
                    // QKV (V portion), gate, conv1d (V portion), and output projection
                    // from grouped→tiled for ggml broadcast consistency.
                    // Therefore ALL tensors use the same tiled head index directly.
                    let mut g_decay = vec![0.0f32; n_v_h];
                    for h in 0..n_v_h {
                        // ssm_a_data already stores -exp(A_log) from GGUF conversion
                        // (convert_hf_to_gguf: data_torch = -torch.exp(data_torch))
                        // Reference: g = -exp(A_log) * softplus(a + dt_bias) = ssm_a * softplus(...)
                        let ssm_a = ssm_a_data[h]; // already = -exp(A_log), tiled order
                        let sp_in = a_raw[h] + ssm_dt_bias[h];
                        let sp = if sp_in > 10.0 {
                            sp_in
                        } else {
                            (1.0_f32 + sp_in.exp()).ln()
                        };
                        g_decay[h] = ssm_a * sp; // negative × positive → negative → exp(g) < 1
                    }

                    // Step 6: Repeat-interleave Q, K from n_k_h → n_v_h heads
                    // GGUF stores V heads in TILED order (reordered by convert_hf_to_gguf.py):
                    //   tiled: [v0_K0, v0_K1, ..., v0_K15, v1_K0, v1_K1, ..., v1_K15]
                    // So V head `vh` in tiled order corresponds to K head `vh % n_k_h`.
                    let mut q_exp = vec![0.0f32; n_v_h * k_hd];
                    let mut k_exp = vec![0.0f32; n_v_h * k_hd];
                    for vh in 0..n_v_h {
                        let kh = vh % n_k_h; // tiled: K head = vh mod n_k_h
                        let src = kh * k_hd;
                        let dst = vh * k_hd;
                        q_exp[dst..dst + k_hd].copy_from_slice(&q_raw[src..src + k_hd]);
                        k_exp[dst..dst + k_hd].copy_from_slice(&k_raw[src..src + k_hd]);
                    }

                    // Step 7: L2-normalize Q and K per head, then scale Q
                    // Reference: fla-org fused_recurrent_gated_delta_rule kernel
                    // Q and K are L2-normalized (ggml_l2_norm in llama.cpp):
                    //   y[i] = x[i] / max(sqrt(sum(x^2)), eps)
                    // then Q is scaled by 1/sqrt(k_hd) for attention scaling.
                    let q_scale = 1.0 / (k_hd as f32).sqrt();
                    for vh in 0..n_v_h {
                        let off = vh * k_hd;
                        let q_slice = &mut q_exp[off..off + k_hd];
                        let norm_q: f32 =
                            q_slice.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                        for x in q_slice.iter_mut() {
                            *x = *x / norm_q * q_scale;
                        }
                        let k_slice = &mut k_exp[off..off + k_hd];
                        let norm_k: f32 =
                            k_slice.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                        for x in k_slice.iter_mut() {
                            *x /= norm_k;
                        }
                    }

                    // Step 8: Delta rule — state [n_v_h, k_hd, v_hd] = [32, 128, 128]
                    let state = &mut self.ssm_states[layer_idx];
                    let mut attn_result = vec![0.0f32; value_dim]; // [4096]
                                                                   // Reusable buffer to avoid per-head allocation
                    let mut kv_mem = vec![0.0f32; v_hd];

                    for vh in 0..n_v_h {
                        let decay = g_decay[vh].exp(); // exp(g) where g < 0 → decay < 1
                        let beta_h = beta_raw[vh];
                        let q_h = &q_exp[vh * k_hd..(vh + 1) * k_hd];
                        let k_h = &k_exp[vh * k_hd..(vh + 1) * k_hd];
                        let v_h = &v_raw[vh * v_hd..(vh + 1) * v_hd];
                        let s_off = vh * k_hd * v_hd;

                        // Decay the state: S *= exp(g)
                        for s in state[s_off..s_off + k_hd * v_hd].iter_mut() {
                            *s *= decay;
                        }

                        // kv_mem[v] = sum_k(S[k,v] * K[k]) for each v
                        kv_mem.fill(0.0);
                        for (ki, &k_val) in k_h.iter().enumerate() {
                            let row_off = s_off + ki * v_hd;
                            for vi in 0..v_hd {
                                kv_mem[vi] += state[row_off + vi] * k_val;
                            }
                        }

                        // Fused: delta rule update + output read (single pass over state)
                        // S += outer(K, (V - kv_mem) * beta), then out[v] = sum_k(S[k,v] * Q[k])
                        let out_h = &mut attn_result[vh * v_hd..(vh + 1) * v_hd];
                        for (ki, (&k_val, &q_val)) in k_h.iter().zip(q_h.iter()).enumerate() {
                            let row_off = s_off + ki * v_hd;
                            for vi in 0..v_hd {
                                let delta = (v_h[vi] - kv_mem[vi]) * beta_h;
                                state[row_off + vi] += k_val * delta;
                                out_h[vi] += state[row_off + vi] * q_val;
                            }
                        }
                    }

                    // Step 9: RMSNormGated — RMSNorm(out) * SiLU(z) per head
                    for vh in 0..n_v_h {
                        let off = vh * v_hd;
                        let out_h = &mut attn_result[off..off + v_hd];
                        let z_h = &z[off..off + v_hd];
                        // RMSNorm
                        let var: f32 = out_h.iter().map(|x| x * x).sum::<f32>() / v_hd as f32;
                        let scale = 1.0 / (var + rms_eps).sqrt();
                        for (i, o) in out_h.iter_mut().enumerate() {
                            let normed_val = *o * scale * ssm_norm_w[i];
                            let z_val = z_h[i];
                            let silu_z = z_val / (1.0 + (-z_val).exp());
                            *o = normed_val * silu_z;
                        }
                    }

                    // Step 10: Output projection [value_dim=4096] → [n_embd=2048]
                    let mut out = vec![0.0f32; n_embd];
                    if !try_gpu!(&mut out, &attn_result, layer_idx, "ssm_out") {
                        crate::quant::matvec_col_major(
                            &mut out,
                            &attn_result,
                            ssm_out_t.raw(),
                            ssm_out_t.qtype,
                            value_dim,
                            n_embd,
                        )?;
                    }
                    out
                } else {
                    // --- Full attention layer ---
                    // Use direct quantized matvec — avoids allocating 234 MB Vec<f32> per weight.
                    let wq_t = self
                        .weights
                        .get(&format!("{prefix}.attn_q.weight"))
                        .ok_or_else(|| anyhow::anyhow!("{prefix}.attn_q.weight not found"))?;
                    let wk_t = self
                        .weights
                        .get(&format!("{prefix}.attn_k.weight"))
                        .ok_or_else(|| anyhow::anyhow!("{prefix}.attn_k.weight not found"))?;
                    let wv_t = self
                        .weights
                        .get(&format!("{prefix}.attn_v.weight"))
                        .ok_or_else(|| anyhow::anyhow!("{prefix}.attn_v.weight not found"))?;
                    let wo_t = self
                        .weights
                        .get(&format!("{prefix}.attn_output.weight"))
                        .ok_or_else(|| anyhow::anyhow!("{prefix}.attn_output.weight not found"))?;

                    // Derive actual KV head dimensions from tensor shapes.
                    // Full attention layers may have different kv_head_dim than GDR layers.
                    // E.g., Qwen3.5-35B-A3B: GDR uses kv_head_dim=128, but full attention
                    // uses kv_head_dim=256 (attn_k.weight [2048, 512] = 2 heads × 256).
                    let actual_kv_total = wk_t
                        .shape
                        .get(1)
                        .copied()
                        .unwrap_or(n_kv_heads * kv_head_dim);
                    let actual_kv_head_dim = actual_kv_total / n_kv_heads.max(1);
                    let actual_v_total = wv_t
                        .shape
                        .get(1)
                        .copied()
                        .unwrap_or(n_kv_heads * v_head_dim);
                    let actual_v_head_dim = actual_v_total / n_kv_heads.max(1);

                    // Q weight packs [query, gate] interleaved per head:
                    // shape [n_embd, n_heads * q_head_dim * 2] where *2 = query + output gate
                    let qg_total = n_heads * q_head_dim * 2;
                    let q_total = n_heads * q_head_dim;

                    let mut qg = vec![0.0f32; qg_total];
                    let mut k = vec![0.0f32; actual_kv_total];
                    let mut v = vec![0.0f32; actual_v_total];

                    if !try_gpu!(&mut qg, normed, layer_idx, "attn_q") {
                        crate::quant::matvec_col_major(
                            &mut qg,
                            normed,
                            wq_t.raw(),
                            wq_t.qtype,
                            n_embd,
                            qg_total,
                        )?;
                    }
                    if !try_gpu!(&mut k, normed, layer_idx, "attn_k") {
                        crate::quant::matvec_col_major(
                            &mut k,
                            normed,
                            wk_t.raw(),
                            wk_t.qtype,
                            n_embd,
                            actual_kv_total,
                        )?;
                    }
                    if !try_gpu!(&mut v, normed, layer_idx, "attn_v") {
                        crate::quant::matvec_col_major(
                            &mut v,
                            normed,
                            wv_t.raw(),
                            wv_t.qtype,
                            n_embd,
                            actual_v_total,
                        )?;
                    }

                    // De-interleave Q and gate: each head stores [q_head_dim Q, q_head_dim gate]
                    let mut q = vec![0.0f32; q_total];
                    let mut output_gate = vec![0.0f32; q_total];
                    for h_idx in 0..n_heads {
                        let src = h_idx * q_head_dim * 2;
                        let dst = h_idx * q_head_dim;
                        q[dst..dst + q_head_dim].copy_from_slice(&qg[src..src + q_head_dim]);
                        output_gate[dst..dst + q_head_dim]
                            .copy_from_slice(&qg[src + q_head_dim..src + 2 * q_head_dim]);
                    }

                    // Q/K norms (optional, present in qwen35moe)
                    if let Ok(q_norm_w) = self
                        .weights
                        .get_data(&format!("{prefix}.attn_q_norm.weight"))
                    {
                        // Apply per-head RMSNorm to the FIRST actual_kv_head_dim dims of each Q head
                        for h in 0..n_heads {
                            let off = h * q_head_dim;
                            // Normalize only the first actual_kv_head_dim elements of each Q head
                            let norm_end = actual_kv_head_dim.min(q_head_dim);
                            let mut qh_norm = q[off..off + norm_end].to_vec();
                            ops::rms_norm(
                                &mut qh_norm,
                                &q[off..off + norm_end],
                                &q_norm_w,
                                rms_eps,
                            );
                            q[off..off + norm_end].copy_from_slice(&qh_norm);
                        }
                    }
                    if let Ok(k_norm_w) = self
                        .weights
                        .get_data(&format!("{prefix}.attn_k_norm.weight"))
                    {
                        for h in 0..n_kv_heads {
                            let off = h * actual_kv_head_dim;
                            let mut kh_norm = k[off..off + actual_kv_head_dim].to_vec();
                            ops::rms_norm(
                                &mut kh_norm,
                                &k[off..off + actual_kv_head_dim],
                                &k_norm_w,
                                rms_eps,
                            );
                            k[off..off + actual_kv_head_dim].copy_from_slice(&kh_norm);
                        }
                    }

                    // Apply RoPE to Q (first actual_kv_head_dim dims of each head) and K
                    // RoPE rotation dim: capped to actual_kv_head_dim for this layer
                    let attn_rope_dim = rope_dim.min(actual_kv_head_dim);

                    // Compute per-head RoPE frequencies for this layer.
                    // Use IMROPE (NeoX pairing) if model has rope_sections, else standard.
                    let attn_freqs: Option<Vec<f32>> = imrope_freqs.as_ref().map(|f| {
                        // IMROPE freqs cover rope_dim/2 pairs; cap to attn_rope_dim/2
                        f[..attn_rope_dim / 2].to_vec()
                    });

                    // Apply RoPE to each Q head and each K head exactly once.
                    // K has fewer heads (n_kv_heads) than Q (n_heads), so we
                    // must not re-rotate K heads when multiple Q heads share one K head.
                    let abs_pos = prev_seq_len + pos;
                    for h in 0..n_heads {
                        let q_off = h * q_head_dim;
                        let mut q_head = q[q_off..q_off + actual_kv_head_dim].to_vec();
                        let kv_h = h % n_kv_heads;
                        let k_off = kv_h * actual_kv_head_dim;

                        if h < n_kv_heads {
                            // First time seeing this K head — rotate both Q and K
                            let mut k_head = k[k_off..k_off + actual_kv_head_dim].to_vec();
                            if let Some(ref freqs) = attn_freqs {
                                ops::apply_rope_with_freqs(
                                    &mut q_head,
                                    &mut k_head,
                                    abs_pos,
                                    actual_kv_head_dim,
                                    freqs,
                                );
                            } else {
                                ops::apply_rope(
                                    &mut q_head,
                                    &mut k_head,
                                    abs_pos,
                                    actual_kv_head_dim,
                                    attn_rope_dim,
                                    rope_freq_base,
                                );
                            }
                            k[k_off..k_off + actual_kv_head_dim].copy_from_slice(&k_head);
                        } else {
                            // K head already rotated — only rotate Q, use dummy for K
                            let mut dummy_k = vec![0.0f32; actual_kv_head_dim];
                            if let Some(ref freqs) = attn_freqs {
                                ops::apply_rope_with_freqs(
                                    &mut q_head,
                                    &mut dummy_k,
                                    abs_pos,
                                    actual_kv_head_dim,
                                    freqs,
                                );
                            } else {
                                ops::apply_rope(
                                    &mut q_head,
                                    &mut dummy_k,
                                    abs_pos,
                                    actual_kv_head_dim,
                                    attn_rope_dim,
                                    rope_freq_base,
                                );
                            }
                        }
                        q[q_off..q_off + actual_kv_head_dim].copy_from_slice(&q_head);
                    }

                    // Append K, V to KV cache (f16 compressed or f32)
                    if let Some(ref mut cc) = cache.compressed {
                        cc.append(&k, &v);
                    } else {
                        cache.key_cache.extend_from_slice(&k);
                        cache.value_cache.extend_from_slice(&v);
                    }

                    let total_seq = prev_seq_len + pos + 1;
                    let attn_out_total = n_heads * actual_kv_head_dim; // output of multi-head attn
                    let mut attn_out = vec![0.0f32; attn_out_total];

                    let use_compressed = cache.compressed.is_some();

                    for h in 0..n_heads {
                        let kv_h = h % n_kv_heads;
                        let q_off = h * q_head_dim;
                        let q_attn = &q[q_off..q_off + actual_kv_head_dim];

                        let mut head_out = vec![0.0f32; actual_kv_head_dim];

                        if use_compressed {
                            // TurboQuant compressed KV cache path
                            let cc = cache.compressed.as_ref().unwrap();
                            let mut scores = cc.attention_scores(q_attn, kv_h);
                            ops::softmax(&mut scores);
                            head_out = cc.weighted_value_sum(&scores, kv_h);
                        } else {
                            // f32 KV cache path with SIMD dot product
                            let scale = 1.0 / (actual_kv_head_dim as f32).sqrt();
                            let mut scores = vec![0.0f32; total_seq];
                            for (t, score) in scores.iter_mut().enumerate() {
                                let kv_idx =
                                    t * n_kv_heads * actual_kv_head_dim + kv_h * actual_kv_head_dim;
                                let k_slice = &cache.key_cache[kv_idx..kv_idx + actual_kv_head_dim];
                                *score = crate::simd::dot_product(q_attn, k_slice) * scale;
                            }
                            ops::softmax(&mut scores);
                            for (t, &w) in scores.iter().enumerate() {
                                let v_idx =
                                    t * n_kv_heads * actual_v_head_dim + kv_h * actual_v_head_dim;
                                let n = actual_kv_head_dim.min(actual_v_head_dim);
                                for (d, ho) in head_out.iter_mut().enumerate().take(n) {
                                    *ho += w * cache.value_cache[v_idx + d];
                                }
                            }
                        }

                        let out_off = h * actual_kv_head_dim;
                        attn_out[out_off..out_off + actual_kv_head_dim].copy_from_slice(&head_out);
                    }

                    // Apply output gate: attn_out *= sigmoid(gate)
                    for i in 0..attn_out_total {
                        let s = 1.0 / (1.0 + (-output_gate[i]).exp());
                        attn_out[i] *= s;
                    }

                    // Output projection: attn_out [n_heads * actual_kv_head_dim] → [n_embd]
                    let mut proj_out = vec![0.0f32; n_embd];
                    if !try_gpu!(&mut proj_out, &attn_out, layer_idx, "attn_output") {
                        crate::quant::matvec_col_major(
                            &mut proj_out,
                            &attn_out,
                            wo_t.raw(),
                            wo_t.qtype,
                            attn_out_total,
                            n_embd,
                        )?;
                    }
                    proj_out
                };

                // Residual: scratch_layer_out[pos] = x[pos] + attn_out
                ops::elem_add(
                    &mut scratch_layer_out[pos_offset..pos_offset + n_embd],
                    &x[pos_offset..pos_offset + n_embd],
                    &attn_out,
                );
            }

            // --- FFN with pre-FFN norm ---
            // In qwen35moe the pre-FFN norm is `post_attention_norm.weight`
            // Fall back to `ffn_norm.weight` for other architectures
            let ffn_norm_key = if self
                .weights
                .has(&format!("{prefix}.post_attention_norm.weight"))
            {
                format!("{prefix}.post_attention_norm.weight")
            } else {
                format!("{prefix}.ffn_norm.weight")
            };
            let ffn_norm_w = self.weights.get_data(&ffn_norm_key)?;

            for pos in 0..n_tokens {
                let pos_offset = pos * n_embd;
                let lo = &scratch_layer_out[pos_offset..pos_offset + n_embd];

                scratch_normed.fill(0.0);
                ops::rms_norm(&mut scratch_normed, lo, &ffn_norm_w, rms_eps);
                let normed = &scratch_normed[..];

                scratch_ffn_out.fill(0.0);
                let ffn_out = &mut scratch_ffn_out[..];

                if self.weights.has(&format!("{prefix}.ffn_gate_exps.weight")) {
                    // --- MoE forward pass ---
                    let moe_cfg = self.config.moe.ok_or_else(|| {
                        anyhow::anyhow!("MoE config missing for layer {layer_idx}")
                    })?;
                    let n_experts = moe_cfg.expert_count;
                    let top_k = moe_cfg.expert_used_count;
                    let expert_ffn_dim = moe_cfg.expert_intermediate_size;

                    // Router: gate_inp.weight [ne0=n_embd, ne1=n_experts] F32
                    let router_t = self
                        .weights
                        .get(&format!("{prefix}.ffn_gate_inp.weight"))
                        .ok_or_else(|| anyhow::anyhow!("{prefix}.ffn_gate_inp.weight not found"))?;
                    let router_raw = router_t.raw();
                    let router_qtype = router_t.qtype;

                    let mut logits = vec![0.0f32; n_experts];
                    crate::quant::matvec_col_major(
                        &mut logits,
                        normed,
                        router_raw,
                        router_qtype,
                        n_embd,
                        n_experts,
                    )?;

                    // Routing weights: softmax probabilities, then select top-k, then renormalize
                    // Reference (Qwen3_5MoeTopKRouter):
                    //   probs = softmax(logits)
                    //   top_k_values, top_k_indices = topk(probs, k)
                    //   weights = top_k_values / sum(top_k_values)

                    // Apply softmax over all experts
                    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_logits: Vec<f32> =
                        logits.iter().map(|&l| (l - max_logit).exp()).collect();
                    let exp_sum: f32 = exp_logits.iter().sum();
                    let probs: Vec<f32> = exp_logits.iter().map(|e| e / exp_sum).collect();

                    // Top-k selection from softmax probs
                    let mut ranked: Vec<usize> = (0..n_experts).collect();
                    ranked.select_nth_unstable_by(top_k - 1, |&a, &b| {
                        probs[b]
                            .partial_cmp(&probs[a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    ranked.truncate(top_k);

                    if let Some(recorder) = &activation_recorder {
                        let mut expert_hits = vec![0.0f32; n_experts];
                        for &expert_idx in &ranked {
                            expert_hits[expert_idx] = 1.0;
                        }
                        recorder.record(layer_idx, &expert_hits);
                    }

                    // Sort experts by index for sequential mmap access pattern
                    ranked.sort_unstable();

                    // Renormalize top-k weights to sum to 1.0
                    let mut sel_logits: Vec<f32> = ranked.iter().map(|&i| probs[i]).collect();
                    let sel_sum: f32 = sel_logits.iter().sum();
                    if sel_sum > 0.0 {
                        for w in sel_logits.iter_mut() {
                            *w /= sel_sum;
                        }
                    }

                    // Compute each selected expert's output — directly on quantized bytes.
                    // No f32 materialisation: ~22 MB per expert saved per call.
                    let gate_t = self
                        .weights
                        .get(&format!("{prefix}.ffn_gate_exps.weight"))
                        .ok_or_else(|| {
                            anyhow::anyhow!("{prefix}.ffn_gate_exps.weight not found")
                        })?;
                    let up_t = self
                        .weights
                        .get(&format!("{prefix}.ffn_up_exps.weight"))
                        .ok_or_else(|| anyhow::anyhow!("{prefix}.ffn_up_exps.weight not found"))?;
                    let down_t = self
                        .weights
                        .get(&format!("{prefix}.ffn_down_exps.weight"))
                        .ok_or_else(|| {
                            anyhow::anyhow!("{prefix}.ffn_down_exps.weight not found")
                        })?;

                    let hot_experts = self
                        .hot_indices_for_layer(layer_idx)
                        .filter(|indices| indices.len() < n_experts)
                        .map(|indices| indices.to_vec());

                    // Extract shared expert data before branching (enables overlap in parallel path)
                    let shared_expert_refs =
                        if self.weights.has(&format!("{prefix}.ffn_gate_shexp.weight")) {
                            let sg = self
                                .weights
                                .get(&format!("{prefix}.ffn_gate_shexp.weight"))
                                .unwrap();
                            let su = self
                                .weights
                                .get(&format!("{prefix}.ffn_up_shexp.weight"))
                                .unwrap();
                            let sd = self
                                .weights
                                .get(&format!("{prefix}.ffn_down_shexp.weight"))
                                .unwrap();
                            let gate_w = self
                                .weights
                                .get_data(&format!("{prefix}.ffn_gate_inp_shexp.weight"))
                                .ok();
                            Some((sg.raw(), su.raw(), sd.raw(), sg.qtype, gate_w))
                        } else {
                            None
                        };

                    let used_hot_cache = hot_experts.is_some();

                    if let Some(hot_experts) = hot_experts {
                        for &expert_idx in &ranked {
                            if hot_experts.binary_search(&expert_idx).is_ok()
                                && !self.hot_expert_cache[layer_idx].contains_key(&expert_idx)
                                && self.hot_expert_cache[layer_idx].len() < top_k
                            {
                                self.hot_expert_cache[layer_idx].insert(
                                    expert_idx,
                                    CachedExpertWeights {
                                        gate: gate_t.expert_to_f32(expert_idx)?,
                                        up: up_t.expert_to_f32(expert_idx)?,
                                        down: down_t.expert_to_f32(expert_idx)?,
                                    },
                                );
                            }
                        }

                        for (rank, &expert_idx) in ranked.iter().enumerate() {
                            let weight = sel_logits[rank];
                            let mut expert_out = vec![0.0f32; n_embd];

                            if let Some(cached) = self.hot_expert_cache[layer_idx].get(&expert_idx)
                            {
                                crate::ops::ffn_swiglu(
                                    &mut expert_out,
                                    normed,
                                    &cached.gate,
                                    &cached.up,
                                    &cached.down,
                                    n_embd,
                                    expert_ffn_dim,
                                );
                            } else {
                                let (gr, _) = gate_t.expert_raw_slice(expert_idx)?;
                                let (ur, _) = up_t.expert_raw_slice(expert_idx)?;
                                let (dr, _) = down_t.expert_raw_slice(expert_idx)?;
                                crate::quant::ffn_swiglu_q(
                                    &mut expert_out,
                                    normed,
                                    gr,
                                    ur,
                                    dr,
                                    gate_t.qtype,
                                    n_embd,
                                    expert_ffn_dim,
                                )?;
                            }

                            for (out, value) in ffn_out.iter_mut().zip(expert_out.iter()) {
                                *out += weight * value;
                            }
                        }
                    } else {
                        // Prefetch ALL selected experts' pages upfront
                        for &eidx in &ranked {
                            if let (Ok((g, _)), Ok((u, _)), Ok((d, _))) = (
                                gate_t.expert_raw_slice(eidx),
                                up_t.expert_raw_slice(eidx),
                                down_t.expert_raw_slice(eidx),
                            ) {
                                prefetch_mmap(g);
                                prefetch_mmap(u);
                                prefetch_mmap(d);
                            }
                        }

                        let qtype = gate_t.qtype;
                        let mid = ranked.len() / 2;
                        let (first_half, second_half) = ranked.split_at(mid);
                        let (w_first, w_second) = sel_logits.split_at(mid);

                        std::thread::scope(|s| -> anyhow::Result<()> {
                            let t1 = s.spawn(|| -> anyhow::Result<Vec<(f32, Vec<f32>)>> {
                                let mut results = Vec::with_capacity(first_half.len());
                                for (i, &expert_idx) in first_half.iter().enumerate() {
                                    let mut expert_out = vec![0.0f32; n_embd];
                                    let (gr, _) = gate_t.expert_raw_slice(expert_idx)?;
                                    let (ur, _) = up_t.expert_raw_slice(expert_idx)?;
                                    let (dr, _) = down_t.expert_raw_slice(expert_idx)?;
                                    crate::quant::ffn_swiglu_q(
                                        &mut expert_out,
                                        normed,
                                        gr,
                                        ur,
                                        dr,
                                        qtype,
                                        n_embd,
                                        expert_ffn_dim,
                                    )?;
                                    results.push((w_first[i], expert_out));
                                }
                                Ok(results)
                            });
                            let t2 = s.spawn(|| -> anyhow::Result<Vec<(f32, Vec<f32>)>> {
                                let mut results = Vec::with_capacity(second_half.len());
                                for (i, &expert_idx) in second_half.iter().enumerate() {
                                    let mut expert_out = vec![0.0f32; n_embd];
                                    let (gr, _) = gate_t.expert_raw_slice(expert_idx)?;
                                    let (ur, _) = up_t.expert_raw_slice(expert_idx)?;
                                    let (dr, _) = down_t.expert_raw_slice(expert_idx)?;
                                    crate::quant::ffn_swiglu_q(
                                        &mut expert_out,
                                        normed,
                                        gr,
                                        ur,
                                        dr,
                                        qtype,
                                        n_embd,
                                        expert_ffn_dim,
                                    )?;
                                    results.push((w_second[i], expert_out));
                                }
                                Ok(results)
                            });

                            // On ≥3 cores: overlap shared expert on main thread while
                            // routed experts run on spawned threads.
                            // On ≤2 cores: skip overlap (would cause contention).
                            let sh_result: Option<(Vec<f32>, f32)> = if cpu_cores >= 3 {
                                if let Some((sg_raw, su_raw, sd_raw, sh_qt, ref gate_w)) =
                                    shared_expert_refs
                                {
                                    let mut sh_out = vec![0.0f32; n_embd];
                                    crate::quant::ffn_swiglu_q(
                                        &mut sh_out,
                                        normed,
                                        sg_raw,
                                        su_raw,
                                        sd_raw,
                                        sh_qt,
                                        n_embd,
                                        expert_ffn_dim,
                                    )?;
                                    let scale = if let Some(ref gw) = *gate_w {
                                        if gw.len() == n_embd {
                                            let dot: f32 = gw
                                                .iter()
                                                .zip(normed.iter())
                                                .map(|(a, b)| a * b)
                                                .sum();
                                            1.0 / (1.0 + (-dot).exp())
                                        } else {
                                            gw.first().copied().unwrap_or(1.0)
                                        }
                                    } else {
                                        1.0
                                    };
                                    Some((sh_out, scale))
                                } else {
                                    None
                                }
                            } else {
                                None
                            };

                            let r1 = t1
                                .join()
                                .map_err(|_| anyhow::anyhow!("thread 1 panicked"))??;
                            let r2 = t2
                                .join()
                                .map_err(|_| anyhow::anyhow!("thread 2 panicked"))??;
                            for (weight, expert_out) in r1.into_iter().chain(r2.into_iter()) {
                                for (o, e) in ffn_out.iter_mut().zip(expert_out.iter()) {
                                    *o += weight * e;
                                }
                            }
                            if let Some((sh_out, scale)) = sh_result {
                                for (o, s) in ffn_out.iter_mut().zip(sh_out.iter()) {
                                    *o += scale * s;
                                }
                            }
                            Ok(())
                        })?;
                    }

                    // Shared expert: runs sequentially when not already overlapped.
                    // Overlapped in parallel path on ≥3 cores; sequential otherwise.
                    if used_hot_cache || cpu_cores < 3 {
                        if let Some((sg_raw, su_raw, sd_raw, sh_qt, ref gate_w)) =
                            shared_expert_refs
                        {
                            let mut sh_out = vec![0.0f32; n_embd];
                            crate::quant::ffn_swiglu_q(
                                &mut sh_out,
                                normed,
                                sg_raw,
                                su_raw,
                                sd_raw,
                                sh_qt,
                                n_embd,
                                expert_ffn_dim,
                            )?;
                            let scale = if let Some(ref gw) = *gate_w {
                                if gw.len() == n_embd {
                                    let dot: f32 =
                                        gw.iter().zip(normed.iter()).map(|(a, b)| a * b).sum();
                                    1.0 / (1.0 + (-dot).exp())
                                } else {
                                    gw.first().copied().unwrap_or(1.0)
                                }
                            } else {
                                1.0
                            };
                            for (o, s) in ffn_out.iter_mut().zip(sh_out.iter()) {
                                *o += scale * s;
                            }
                        }
                    }
                } else {
                    // Dense FFN — quantized path (no f32 materialisation)
                    let ffn_gate_t = self
                        .weights
                        .get(&format!("{prefix}.ffn_gate.weight"))
                        .ok_or_else(|| anyhow::anyhow!("{prefix}.ffn_gate.weight not found"))?;
                    let ffn_up_t = self
                        .weights
                        .get(&format!("{prefix}.ffn_up.weight"))
                        .ok_or_else(|| anyhow::anyhow!("{prefix}.ffn_up.weight not found"))?;
                    let ffn_down_t = self
                        .weights
                        .get(&format!("{prefix}.ffn_down.weight"))
                        .ok_or_else(|| anyhow::anyhow!("{prefix}.ffn_down.weight not found"))?;
                    let n_ff = self.config.feed_forward_length;

                    if let Some(recorder) = &activation_recorder {
                        self.record_dense_ffn_activations(
                            recorder,
                            layer_idx,
                            normed,
                            ffn_gate_t.raw(),
                            ffn_up_t.raw(),
                            ffn_gate_t.qtype,
                            n_embd,
                            n_ff,
                        )?;
                    }

                    if let Some(hot_indices) = self.sparse_hot_indices(layer_idx, n_ff) {
                        crate::quant::ffn_swiglu_q_selected(
                            ffn_out,
                            normed,
                            ffn_gate_t.raw(),
                            ffn_up_t.raw(),
                            ffn_down_t.raw(),
                            ffn_gate_t.qtype,
                            n_embd,
                            n_ff,
                            hot_indices,
                        )?;
                    } else {
                        crate::quant::ffn_swiglu_q(
                            ffn_out,
                            normed,
                            ffn_gate_t.raw(),
                            ffn_up_t.raw(),
                            ffn_down_t.raw(),
                            ffn_gate_t.qtype,
                            n_embd,
                            n_ff,
                        )?;
                    }
                }

                // Residual: in-place add ffn_out to scratch_layer_out
                for (o, &f) in scratch_layer_out[pos_offset..pos_offset + n_embd]
                    .iter_mut()
                    .zip(ffn_out.iter())
                {
                    *o += f;
                }
            }

            std::mem::swap(&mut x, &mut scratch_layer_out);

            // Per-layer diagnostics
            if diagnostics_enabled() {
                let last_off = (n_tokens - 1) * n_embd;
                let h = &x[last_off..last_off + n_embd];
                let norm: f32 = h.iter().map(|x| x * x).sum::<f32>().sqrt();
                let max_abs: f32 = h.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let layer_type = if is_ssm { "SSM" } else { "ATN" };
                let layer_ms = layer_start.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[LAYER {layer_idx:>2}] {layer_type} norm={norm:.2} max={max_abs:.4} time={layer_ms:.1}ms");
            }
        }

        // --- Final norm + LM head ---
        let lm_start = std::time::Instant::now();
        let output_norm_w = self.weights.get_data("output_norm.weight")?;
        let mut final_x = vec![0.0f32; n_embd];
        ops::rms_norm(
            &mut final_x,
            &x[(n_tokens - 1) * n_embd..],
            &output_norm_w,
            rms_eps,
        );

        // LM head: output.weight [ne0=n_embd, ne1=n_vocab] — use quantized col-major matmul
        let lm_head_tensor = self
            .weights
            .get("output.weight")
            .or_else(|| self.weights.get("token_embd.weight"))
            .ok_or_else(|| anyhow::anyhow!("output.weight not found"))?;
        let actual_vocab = lm_head_tensor.shape.get(1).copied().unwrap_or(n_vocab);

        let mut logits = vec![0.0f32; actual_vocab];

        // Try GPU LM head first (split across GPUs for parallel computation)
        #[allow(unused_mut)]
        let mut used_gpu = false;
        #[cfg(feature = "cuda")]
        {
            if let Some(ref gpu_ref) = self.gpu {
                if let Some(result) = gpu_ref.try_lm_head_matvec(&mut logits, &final_x) {
                    result.map_err(|e| anyhow::anyhow!("GPU LM head: {e}"))?;
                    used_gpu = true;
                }
            }
        }
        if !used_gpu {
            crate::quant::matvec_col_major(
                &mut logits,
                &final_x,
                lm_head_tensor.raw(),
                lm_head_tensor.qtype,
                n_embd,
                actual_vocab,
            )?;
        }

        if n_tokens == 1 && diagnostics_enabled() {
            let lm_ms = lm_start.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[TIMING] LM head: {lm_ms:.1}ms");
        }

        Ok(logits)
    }

    /// Get model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get backend information
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }

    /// Get tokenizer reference
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get weights reference
    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// Get hardware profile (computed at startup)
    pub fn hardware_profile(&self) -> Option<&HardwareProfile> {
        self.hw_profile.as_ref()
    }

    /// Get execution plan (computed at startup)
    pub fn execution_plan(&self) -> Option<&ExecutionPlan> {
        self.exec_plan.as_ref()
    }

    /// Configure an activation recorder for profiling.
    pub fn set_activation_recorder(&mut self, recorder: ActivationRecorder) {
        self.activation_recorder = Some(recorder);
    }

    /// Remove the activation recorder after profiling.
    pub fn clear_activation_recorder(&mut self) {
        self.activation_recorder = None;
    }

    /// Load a profiler-generated hot index for sparse dense FFN execution.
    pub fn set_hot_index(&mut self, mut hot_index: HotNeuronIndex) -> Result<()> {
        for layer in &mut hot_index.layers {
            layer.hot_indices.sort_unstable();
            layer.hot_indices.dedup();
        }

        hot_index.validate_against_dims(&self.profiling_layer_dims(None))?;

        let mut plan = vec![None; self.config.block_count];
        for layer in hot_index.layers {
            plan[layer.layer_idx] = Some(layer.hot_indices);
        }

        self.hot_index = Some(plan);
        for cache in &mut self.hot_expert_cache {
            cache.clear();
        }
        Ok(())
    }

    /// Remove any loaded sparse execution plan.
    pub fn clear_hot_index(&mut self) {
        self.hot_index = None;
        for cache in &mut self.hot_expert_cache {
            cache.clear();
        }
    }

    /// Determine the activation width recorded for each profiled layer.
    pub fn profiling_layer_dims(&self, layer_limit: Option<usize>) -> Vec<usize> {
        let limit = layer_limit
            .unwrap_or(self.config.block_count)
            .min(self.config.block_count);

        (0..limit)
            .map(|layer_idx| {
                let prefix = format!("blk.{layer_idx}");
                if self.weights.has(&format!("{prefix}.ffn_gate_exps.weight")) {
                    self.config
                        .moe
                        .map(|moe| moe.expert_count)
                        .unwrap_or(self.config.feed_forward_length)
                } else {
                    self.weights
                        .get(&format!("{prefix}.ffn_gate.weight"))
                        .and_then(|tensor| tensor.shape.get(1).copied())
                        .unwrap_or(self.config.feed_forward_length)
                }
            })
            .collect()
    }

    fn hot_indices_for_layer(&self, layer_idx: usize) -> Option<&[usize]> {
        self.hot_index.as_ref()?.get(layer_idx)?.as_deref()
    }

    fn sparse_hot_indices(&self, layer_idx: usize, dense_width: usize) -> Option<&[usize]> {
        let indices = self.hot_indices_for_layer(layer_idx)?;
        (indices.len() < dense_width).then_some(indices)
    }

    /// Enable TurboQuant compressed KV cache
    /// Call before generation. Existing caches are cleared.
    pub fn enable_compressed_cache(&mut self) {
        if self.use_compressed_cache {
            return;
        }
        let n_kv_heads = self
            .config
            .attention
            .head_count_kv
            .unwrap_or(self.config.attention.head_count);
        let kv_head_dim = self.config.attention.kv_head_dim();
        self.layer_caches = (0..self.config.block_count)
            .map(|layer_idx| {
                let layer_kv_dim = self
                    .weights
                    .get(&format!("blk.{layer_idx}.attn_k.weight"))
                    .and_then(|t| t.shape.get(1).copied())
                    .map(|total| total / n_kv_heads.max(1))
                    .unwrap_or(kv_head_dim);
                LayerCache::new(true, n_kv_heads, layer_kv_dim)
            })
            .collect();
        self.use_compressed_cache = true;
    }

    /// Get KV cache memory usage (f32 bytes)
    pub fn kv_cache_memory_bytes(&self) -> usize {
        self.layer_caches
            .iter()
            .map(|c| (c.key_cache.len() + c.value_cache.len()) * 4)
            .sum()
    }

    /// Get compressed KV cache memory usage (if enabled)
    pub fn compressed_cache_memory_bytes(&self) -> usize {
        self.layer_caches
            .iter()
            .filter_map(|c| c.compressed.as_ref().map(|cc| cc.memory_bytes()))
            .sum()
    }

    /// Reset KV caches, SSM states, and conv states
    pub fn reset(&mut self) {
        let n_kv_heads = self
            .config
            .attention
            .head_count_kv
            .unwrap_or(self.config.attention.head_count);
        let kv_head_dim = self.config.attention.kv_head_dim();
        for (layer_idx, cache) in self.layer_caches.iter_mut().enumerate() {
            cache.key_cache.clear();
            cache.value_cache.clear();
            if self.use_compressed_cache {
                let layer_kv_dim = self
                    .weights
                    .get(&format!("blk.{layer_idx}.attn_k.weight"))
                    .and_then(|t| t.shape.get(1).copied())
                    .map(|total| total / n_kv_heads.max(1))
                    .unwrap_or(kv_head_dim);
                cache.compressed = Some(CompressedKVCache::new(n_kv_heads, layer_kv_dim));
            }
        }
        for state in &mut self.ssm_states {
            state.fill(0.0);
        }
        for conv_state in &mut self.ssm_conv_states {
            conv_state.clear();
        }
    }

    /// Return (layer_idx, frobenius_norm) for each SSM layer's state
    pub fn ssm_state_norms(&self) -> Vec<(usize, f32)> {
        self.ssm_states
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.is_empty())
            .map(|(i, s)| {
                let norm: f32 = s.iter().map(|x| x * x).sum::<f32>().sqrt();
                (i, norm)
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn record_dense_ffn_activations(
        &self,
        recorder: &ActivationRecorder,
        layer_idx: usize,
        normed: &[f32],
        gate_raw: &[u8],
        up_raw: &[u8],
        qtype: QuantizationType,
        n_embd: usize,
        n_ff: usize,
    ) -> Result<()> {
        let mut gate = vec![0.0f32; n_ff];
        let mut up = vec![0.0f32; n_ff];

        crate::quant::matvec_col_major(&mut gate, normed, gate_raw, qtype, n_embd, n_ff)?;
        crate::quant::matvec_col_major(&mut up, normed, up_raw, qtype, n_embd, n_ff)?;

        for i in 0..n_ff {
            let sig = 1.0 / (1.0 + (-gate[i]).exp());
            gate[i] = gate[i] * sig * up[i];
        }

        recorder.record(layer_idx, &gate);
        Ok(())
    }
}

fn diagnostics_enabled() -> bool {
    matches!(
        std::env::var("POWERINFER_TRACE_TOKENS").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

/// Greedy sampling: pick the token with highest logit
fn sample_argmax(logits: &[f32]) -> u32 {
    let mut best_id = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &val) in logits.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_id = i as u32;
        }
    }
    best_id
}

/// Apply repetition penalty to logits for tokens that have already been generated.
/// Penalty > 1.0 reduces the probability of repeating; 1.0 = no effect.
/// Uses the standard approach: divide positive logits by penalty, multiply negative ones.
fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    if penalty <= 1.0 || !penalty.is_finite() {
        return;
    }
    for &token_id in generated {
        let idx = token_id as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

fn sample_token(logits: &[f32], temperature: f32, top_p: f32, rng: &mut SplitMix64) -> u32 {
    if !temperature.is_finite() || temperature <= 0.0 {
        return sample_argmax(logits);
    }

    let temperature = temperature.max(1e-5);
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut weights = Vec::with_capacity(logits.len());
    let mut total = 0.0f64;

    for &logit in logits {
        let weight = ((logit - max_logit) / temperature).exp() as f64;
        if weight.is_finite() {
            total += weight;
            weights.push(weight);
        } else {
            weights.push(0.0);
        }
    }

    if total <= 0.0 || !total.is_finite() {
        return sample_argmax(logits);
    }

    let top_p = if top_p.is_finite() {
        top_p.clamp(0.0, 1.0)
    } else {
        1.0
    };

    if top_p >= 0.999_999 {
        return sample_from_weight_slice(&weights, total, rng);
    }

    let mut ranked: Vec<(usize, f64)> = weights.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut truncated = Vec::new();
    let mut cumulative = 0.0f64;
    for (idx, weight) in ranked {
        if weight <= 0.0 {
            continue;
        }
        cumulative += weight / total;
        truncated.push((idx, weight));
        if cumulative >= top_p as f64 {
            break;
        }
    }

    if truncated.is_empty() {
        return sample_argmax(logits);
    }

    let truncated_total: f64 = truncated.iter().map(|(_, weight)| weight).sum();
    let draw = rng.next_f64() * truncated_total;
    let mut acc = 0.0f64;
    for (idx, weight) in truncated {
        acc += weight;
        if draw <= acc {
            return idx as u32;
        }
    }

    sample_argmax(logits)
}

fn sample_from_weight_slice(weights: &[f64], total: f64, rng: &mut SplitMix64) -> u32 {
    let draw = rng.next_f64() * total;
    let mut acc = 0.0f64;

    for (idx, weight) in weights.iter().copied().enumerate() {
        acc += weight;
        if draw <= acc {
            return idx as u32;
        }
    }

    sample_argmax(
        &weights
            .iter()
            .map(|weight| *weight as f32)
            .collect::<Vec<_>>(),
    )
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Prefetch mmap pages into OS page cache using madvise(MADV_WILLNEED).
/// Non-blocking: tells the kernel to asynchronously read pages ahead.
fn prefetch_mmap(data: &[u8]) {
    if data.is_empty() {
        return;
    }
    #[cfg(target_os = "linux")]
    {
        // SAFETY: data is a valid memory-mapped region; MADV_WILLNEED is advisory only.
        unsafe {
            libc::madvise(
                data.as_ptr() as *mut libc::c_void,
                data.len(),
                libc::MADV_WILLNEED,
            );
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        // Fallback: touch one byte per page to trigger read-ahead
        const PAGE_SIZE: usize = 4096;
        let mut _sum: u8 = 0;
        for offset in (0..data.len()).step_by(PAGE_SIZE) {
            _sum = _sum.wrapping_add(data[offset]);
        }
        std::hint::black_box(_sum);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_model_config_parsing() {
        let path = "/home/jon/models/llama-cache/Arch-Agent-3B.Q8_0.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("SKIP: model not found");
            return;
        }
        let gguf = crate::gguf::GgufFile::open(path).expect("GGUF should load");
        let config = gguf.model_config().expect("config should parse");
        assert_eq!(config.arch, "qwen2");
        assert_eq!(config.block_count, 36);
        assert_eq!(config.embedding_length, 2048);
        assert_eq!(config.attention.head_count, 16);
        assert_eq!(config.attention.head_dim, 128);
    }
}
