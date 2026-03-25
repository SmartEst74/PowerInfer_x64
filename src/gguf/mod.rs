//! GGUF file format parser and model loader
//!
//! Based on the GGUF specification v2/v3.
//! Uses `gguf-rs` crate for low-level parsing.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use anyhow::{anyhow, Result};

use gguf_rs::{GgufFile as GgufRsFile, TensorInfo, Value};

use crate::quant::{QuantizationType, dequantize_block};
use crate::model::{ModelConfig, LayerConfig};

/// Higher-level wrapper around `gguf-rs`'s GgufFile
pub struct GgufFile {
    inner: GgufRsFile<BufReader<File>>,
    path: std::path::PathBuf,
}

impl GgufFile {
    /// Open a GGUF file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let inner = GgufRsFile::parse(reader)?;
        Ok(Self {
            inner,
            path: path.as_ref().to_path_buf(),
        })
    }

    /// Get model architecture name (e.g., "qwen3", "llama")
    pub fn architecture(&self) -> Result<&str> {
        self.inner
            .metadata()
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Architecture not found in GGUF metadata"))
    }

    /// Get model name
    pub fn name(&self) -> Option<&str> {
        self.inner.metadata().get("general.name").and_then(|v| v.as_str())
    }

    /// Get number of model parameters (as string like "7B")
    pub fn parameter_count(&self) -> Option<&str> {
        self.inner.metadata().get("general.parameter_count").and_then(|v| v.as_str())
    }

    /// Get context length
    pub fn context_length(&self) -> Result<usize> {
        self.inner.metadata()
            .get("llama.context_length")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .ok_or_else(|| anyhow!("Context length not found"))
    }

    /// Get embedding dimension
    pub fn embedding_length(&self) -> Result<usize> {
        self.inner.metadata()
            .get("llama.embedding_length")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .ok_or_else(|| anyhow!("Embedding length not found"))
    }

    /// Get number of transformer layers
    pub fn block_count(&self) -> Result<usize> {
        self.inner.metadata()
            .get("llama.block_count")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .ok_or_else(|| anyhow!("Block count not found"))
    }

    /// Get number of attention heads
    pub fn attention_head_count(&self) -> Result<usize> {
        self.inner.metadata()
            .get("llama.attention.head_count")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .ok_or_else(|| anyhow!("Attention head count not found"))
    }

    /// Get number of key/value heads (for GQA)
    pub fn attention_head_count_kv(&self) -> Option<usize> {
        self.inner.metadata()
            .get("llama.attention.head_count_kv")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
    }

    /// Get feed forward length (intermediate size)
    pub fn feed_forward_length(&self) -> Result<usize> {
        self.inner.metadata()
            .get("llama.feed_forward_length")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .ok_or_else(|| anyhow!("Feed forward length not found"))
    }

    /// Get RoPE dimension (partial rotary embedding dimension)
    pub fn rope_dim(&self) -> Option<usize> {
        self.inner.metadata()
            .get("llama.rope.dimension")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
    }

    /// Get MoE configuration (if present)
    pub fn moe_config(&self) -> Option<MoeConfig> {
        let expert_count = self.inner.metadata()
            .get("llama.expert_count")
            .and_then(|v| v.as_u64())? as usize;
        let expert_used_count = self.inner.metadata()
            .get("llama.expert_used_count")
            .and_then(|v| v.as_u64())? as usize;
        let expert_intermediate_size = self.inner.metadata()
            .get("llama.expert_feed_forward_length")
            .and_then(|v| v.as_u64())? as usize;

        Some(MoeConfig {
            expert_count,
            expert_used_count,
            expert_intermediate_size,
        })
    }

    /// Get Qwen3-specific: full attention interval (for hybrid DeltaNet+Full)
    pub fn qwen_full_attention_interval(&self) -> Option<usize> {
        self.inner.metadata()
            .get("qwen3.full_attention_interval")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
    }

    /// Build model configuration structure
    pub fn model_config(&self) -> Result<ModelConfig> {
        Ok(ModelConfig {
            arch: self.architecture()?.to_string(),
            name: self.name().map(String::from),
            context_length: self.context_length()?,
            embedding_length: self.embedding_length()?,
            block_count: self.block_count()?,
            attention: LayerConfig {
                head_count: self.attention_head_count()?,
                head_count_kv: self.attention_head_count_kv(),
                head_dim: self.embedding_length()? / self.attention_head_count()?,
                rope_dim: self.rope_dim(),
                full_attention_interval: self.qwen_full_attention_interval(),
            },
            feed_forward_length: self.feed_forward_length()?,
            moe: self.moe_config(),
            quantization: self.quantization_type()?,
        })
    }

    /// Get quantization type from first tensor
    fn quantization_type(&self) -> Result<QuantizationType> {
        let first_tensor = self.inner.tensors().first()
            .ok_or_else(|| anyhow!("No tensors in GGUF file"))?;
        QuantizationType::from_ggml_type(first_tensor.data_type)
    }

    /// Get path to the underlying file (for mmap)
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get raw tensor info
    pub fn tensors(&self) -> &[TensorInfo] {
        self.inner.tensors()
    }

    /// Get metadata value by key
    pub fn metadata(&self, key: &str) -> Option<&Value> {
        self.inner.metadata().get(key)
    }
}

/// Mixture-of-Experts configuration
#[derive(Debug, Clone, Copy)]
pub struct MoeConfig {
    pub expert_count: usize,
    pub expert_used_count: usize,
    pub expert_intermediate_size: usize,
}

/// Quantization type mapping from GGML
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2_K,
    Q3_K_S,
    Q3_K_M,
    Q3_K_L,
    Q4_K_S,
    Q4_K_M,
    Q5_K_S,
    Q5_K_M,
    Q6_K,
    IQ1_S,
    IQ1_M,
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,
    IQ3_XXS,
    IQ3_S,
    IQ3_M,
    IQ4_NL,
    IQ4_XS,
    F16,
    F32,
}

impl QuantizationType {
    fn from_ggml_type(ty: gguf_rs::GgmlType) -> Result<Self> {
        use gguf_rs::GgmlType::*;
        match ty {
            F32 => Ok(Self::F32),
            F16 => Ok(Self::F16),
            Q4_0 => Ok(Self::Q4_0),
            Q4_1 => Ok(Self::Q4_1),
            Q5_0 => Ok(Self::Q5_0),
            Q5_1 => Ok(Self::Q5_1),
            Q8_0 => Ok(Self::Q8_0),
            Q2_K => Ok(Self::Q2_K),
            Q3_K_S => Ok(Self::Q3_K_S),
            Q3_K_M => Ok(Self::Q3_K_M),
            Q3_K_L => Ok(Self::Q3_K_L),
            Q4_K_S => Ok(Self::Q4_K_S),
            Q4_K_M => Ok(Self::Q4_K_M),
            Q5_K_S => Ok(Self::Q5_K_S),
            Q5_K_M => Ok(Self::Q5_K_M),
            Q6_K => Ok(Self::Q6_K),
            IQ1_S => Ok(Self::IQ1_S),
            IQ1_M => Ok(Self::IQ1_M),
            IQ2_XXS => Ok(Self::IQ2_XXS),
            IQ2_XS => Ok(Self::IQ2_XS),
            IQ2_S => Ok(Self::IQ2_S),
            IQ3_XXS => Ok(Self::IQ3_XXS),
            IQ3_S => Ok(Self::IQ3_S),
            IQ3_M => Ok(Self::IQ3_M),
            IQ4_NL => Ok(Self::IQ4_NL),
            IQ4_XS => Ok(Self::IQ4_XS),
            _ => Err(anyhow!("Unsupported GGML type: {:?}", ty)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_gguf() {
        // Integration test will load actual file
    }
}
