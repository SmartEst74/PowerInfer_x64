//! GGUF file format parser and model loader
//!
//! Based on the GGUF specification v2/v3.
//! Uses `gguf-rs` crate for low-level parsing.

use anyhow::{anyhow, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

use gguf_rs::{GGMLType, GGUFModel, Tensor};

use crate::model::{LayerConfig, ModelConfig};
use crate::quant::QuantizationType;

/// Higher-level wrapper around `gguf-rs`'s GGUFModel
pub struct GgufFile {
    model: GGUFModel,
    path: std::path::PathBuf,
}

impl GgufFile {
    /// Open a GGUF file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| anyhow!("Invalid path"))?;
        let mut container = gguf_rs::get_gguf_container_array_size(path_str, u64::MAX)?;
        let model = container.decode()?;
        Ok(Self {
            model,
            path: path.as_ref().to_path_buf(),
        })
    }

    /// Get metadata map
    fn kv(&self) -> &BTreeMap<String, Value> {
        self.model.metadata()
    }

    /// Look up a metadata key, trying architecture-specific prefix first, then llama.
    fn get_config(&self, suffix: &str) -> Option<&Value> {
        let arch = self.architecture().ok()?;
        let arch_key = format!("{arch}.{suffix}");
        self.kv()
            .get(&arch_key)
            .or_else(|| self.kv().get(&format!("llama.{suffix}")))
    }

    /// Get model architecture name (e.g., "qwen3", "llama")
    pub fn architecture(&self) -> Result<&str> {
        self.kv()
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Architecture not found in GGUF metadata"))
    }

    /// Get model name
    pub fn name(&self) -> Option<&str> {
        self.kv().get("general.name").and_then(|v| v.as_str())
    }

    /// Get number of model parameters (as string like "7B")
    pub fn parameter_count(&self) -> Option<&str> {
        self.kv()
            .get("general.parameter_count")
            .and_then(|v| v.as_str())
    }

    /// Get context length
    pub fn context_length(&self) -> Result<usize> {
        self.get_config("context_length")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .ok_or_else(|| anyhow!("Context length not found"))
    }

    /// Get embedding dimension
    pub fn embedding_length(&self) -> Result<usize> {
        self.get_config("embedding_length")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .ok_or_else(|| anyhow!("Embedding length not found"))
    }

    /// Get number of transformer layers
    pub fn block_count(&self) -> Result<usize> {
        self.get_config("block_count")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .ok_or_else(|| anyhow!("Block count not found"))
    }

    /// Get number of attention heads
    pub fn attention_head_count(&self) -> Result<usize> {
        self.get_config("attention.head_count")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .ok_or_else(|| anyhow!("Attention head count not found"))
    }

    /// Get number of key/value heads (for GQA)
    pub fn attention_head_count_kv(&self) -> Option<usize> {
        self.get_config("attention.head_count_kv")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
    }

    /// Get feed forward length (intermediate size)
    pub fn feed_forward_length(&self) -> Result<usize> {
        self.get_config("feed_forward_length")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .or_else(|| {
                // MoE models may not have feed_forward_length — use expert intermediate size
                self.moe_config().map(|m| m.expert_intermediate_size)
            })
            .ok_or_else(|| anyhow!("Feed forward length not found"))
    }

    /// Get attention key (query) head dimension
    pub fn attention_key_length(&self) -> Option<usize> {
        self.get_config("attention.key_length")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
    }

    /// Get attention value head dimension
    pub fn attention_value_length(&self) -> Option<usize> {
        self.get_config("attention.value_length")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
    }

    /// Get RoPE rotation dimension count
    pub fn rope_dim(&self) -> Option<usize> {
        // qwen35moe uses rope.dimension_count; older models use rope.dimension
        self.get_config("rope.dimension_count")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .or_else(|| {
                self.get_config("rope.dimension")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize)
            })
    }

    /// Get MoE configuration (if present)
    pub fn moe_config(&self) -> Option<MoeConfig> {
        let expert_count = self.get_config("expert_count").and_then(|v| v.as_u64())? as usize;
        let expert_used_count = self
            .get_config("expert_used_count")
            .and_then(|v| v.as_u64())? as usize;

        // expert_feed_forward_length may not be in metadata — derive from tensor shape
        let expert_intermediate_size = self
            .get_config("expert_feed_forward_length")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .or_else(|| {
                // Derive from ffn_gate_exps.weight shape: [n_embd, expert_ffn_dim, n_experts, 1]
                self.tensors()
                    .iter()
                    .find(|t| t.name.contains("ffn_gate_exps.weight"))
                    .and_then(|t| {
                        if t.shape.len() >= 2 {
                            Some(t.shape[1] as usize)
                        } else {
                            None
                        }
                    })
            })?;

        Some(MoeConfig {
            expert_count,
            expert_used_count,
            expert_intermediate_size,
        })
    }

    /// SSM (Mamba-2) configuration for hybrid architectures
    pub fn ssm_config(&self) -> Option<SsmConfig> {
        let inner_size = self.get_config("ssm.inner_size").and_then(|v| v.as_u64())? as usize;
        let state_size = self.get_config("ssm.state_size").and_then(|v| v.as_u64())? as usize;
        let conv_kernel = self.get_config("ssm.conv_kernel").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
        let time_step_rank = self.get_config("ssm.time_step_rank").and_then(|v| v.as_u64()).unwrap_or(32) as usize;
        let group_count = self.get_config("ssm.group_count").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        Some(SsmConfig { inner_size, state_size, conv_kernel, time_step_rank, group_count })
    }

    /// Get full attention interval (if present; controls which layers use full vs linear attention)
    pub fn qwen_full_attention_interval(&self) -> Option<usize> {
        // Try architecture-specific key first
        self.get_config("full_attention_interval")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
    }

    /// Build model configuration structure
    pub fn model_config(&self) -> Result<ModelConfig> {
        let n_heads = self.attention_head_count()?;
        let n_embd = self.embedding_length()?;
        // Prefer explicit key_length (e.g. qwen35moe), else derive from embedding/heads
        let head_dim = self.attention_key_length()
            .unwrap_or_else(|| n_embd / n_heads);
        Ok(ModelConfig {
            arch: self.architecture()?.to_string(),
            name: self.name().map(String::from),
            context_length: self.context_length()?,
            embedding_length: n_embd,
            block_count: self.block_count()?,
            attention: LayerConfig {
                head_count: n_heads,
                head_count_kv: self.attention_head_count_kv(),
                head_dim,
                key_length: self.attention_key_length(),
                value_length: self.attention_value_length(),
                rope_dim: self.rope_dim(),
                full_attention_interval: self.qwen_full_attention_interval(),
            },
            feed_forward_length: self.feed_forward_length()?,
            moe: self.moe_config(),
            ssm: self.ssm_config(),
            quantization: self.quantization_type()?,
            rms_epsilon: self
                .get_config("attention.layer_norm_rms_epsilon")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-6) as f32,
            rope_freq_base: self
                .get_config("rope.freq_base")
                .and_then(|v| v.as_f64())
                .unwrap_or(10000.0) as f32,
        })
    }

    /// Get quantization type from first tensor
    fn quantization_type(&self) -> Result<QuantizationType> {
        let first_tensor = self
            .model
            .tensors()
            .first()
            .ok_or_else(|| anyhow!("No tensors in GGUF file"))?;
        let ggml_type: GGMLType = first_tensor
            .kind
            .try_into()
            .map_err(|_| anyhow!("Failed to parse GGML type"))?;
        QuantizationType::from_ggml_type(ggml_type)
    }

    /// Get path to the underlying file (for mmap)
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get raw tensor info
    pub fn tensors(&self) -> &[Tensor] {
        self.model.tensors()
    }

    /// Get metadata value by key
    pub fn metadata(&self, key: &str) -> Option<&Value> {
        self.kv().get(key)
    }
}

/// Mixture-of-Experts configuration
#[derive(Debug, Clone, Copy)]
pub struct MoeConfig {
    pub expert_count: usize,
    pub expert_used_count: usize,
    pub expert_intermediate_size: usize,
}

/// SSM (Mamba-2 / Gated DeltaNet) configuration for hybrid architectures
#[derive(Debug, Clone, Copy)]
pub struct SsmConfig {
    /// Inner state dimension (d_inner)
    pub inner_size: usize,
    /// State size for selective scan (d_state)
    pub state_size: usize,
    /// Conv1D kernel size
    pub conv_kernel: usize,
    /// Time step projection rank
    pub time_step_rank: usize,
    /// Number of scan groups
    pub group_count: usize,
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_load_gguf() {
        let path = "/home/jon/models/llama-cache/Arch-Agent-3B.Q8_0.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("SKIP: model not found");
            return;
        }
        let gguf = super::GgufFile::open(path).expect("GGUF should load");
        let config = gguf.model_config().expect("config should parse");
        assert_eq!(config.arch, "qwen2");
        assert_eq!(config.block_count, 36);
        assert_eq!(config.embedding_length, 2048);
    }
}
