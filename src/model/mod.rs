//! Model architectures and inference logic

use std::path::Path;

use crate::gguf::{GgufFile, ModelConfig};
use crate::quant::QuantizationType;
use crate::runtime::{Backend, BackendFactory};
use crate::Result;

/// Model configuration extracted from GGUF
#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub head_count: usize,
    pub head_count_kv: Option<usize>,
    pub head_dim: usize,
    pub rope_dim: Option<usize>,
    pub full_attention_interval: Option<usize>,  // For hybrid DeltaNet
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
    pub quantization: QuantizationType,
}

/// Inference context - main entry point for generation
pub struct InferenceContext {
    config: ModelConfig,
    backend: Box<dyn Backend>,
    // tokenizer: Tokenizer,
    // weights: WeightLoader,
    // kv_cache: KVCache,
    // other state...
}

impl InferenceContext {
    /// Build inference context from GGUF file
    pub fn from_gguf<P: AsRef<Path>>(
        gguf_path: P,
        backend: Backend,
    ) -> Result<Self> {
        let gguf = GgufFile::open(gguf_path)?;
        let config = gguf.model_config()?;

        // TODO: Validate backend supports model architecture
        // TODO: Load weights into backend memory
        // TODO: Initialize tokenizer
        // TODO: Allocate KV cache

        Ok(Self {
            config,
            backend,
        })
    }

    /// Generate text given a prompt
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        // TODO: Tokenize prompt
        // TODO: Prefill: run forward pass on entire prompt
        // TODO: Decoding loop: sample next token, append, repeat
        // TODO: Use backend for all tensor operations
        Ok(format!("Generated text for: {}", prompt))
    }

    /// Get model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get backend information
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_parsing() {
        // Will test with actual GGUF file in integration
    }
}
