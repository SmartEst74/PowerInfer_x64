//! Tokenizer for model vocabulary
//!
//! Currently provides BPE tokenizer compatible with GPT-2 style tokenizers.
//! Loads vocab and merges from GGUF file.

use std::collections::HashMap;

use crate::gguf::GgufFile;
use anyhow::anyhow;

/// Simple BPE tokenizer
pub struct Tokenizer {
    /// Token ID → string
    vocab: Vec<String>,
    /// Pair of token IDs → merged token ID
    merges: HashMap<(u32, u32), u32>,
    /// Special token IDs
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
}

impl Tokenizer {
    /// Load tokenizer from GGUF metadata
    pub fn from_gguf(gguf: &GgufFile) -> crate::Result<Self> {
        // Extract vocabulary from GGUF
        let token_count = gguf
            .metadata("tokenizer.ggml.token_count")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("Missing token count"))? as usize;

        // In real implementation, load all token strings from GGUF
        // For now, placeholder
        let mut vocab = Vec::with_capacity(token_count);
        for i in 0..token_count {
            vocab.push(format!("<token_{}>", i));
        }

        // Load merges if present
        let merges = HashMap::new(); // Placeholder

        // Special tokens
        let bos_token_id = gguf
            .metadata("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);
        let eos_token_id = gguf
            .metadata("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);
        let pad_token_id = gguf
            .metadata("tokenizer.ggml.pad_token_id")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);

        Ok(Self {
            vocab,
            merges,
            bos_token_id,
            eos_token_id,
            pad_token_id,
        })
    }

    /// Encode a string into token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Placeholder: just hash to determine tokens
        let mut ids = Vec::new();
        for ch in text.chars() {
            let id = (ch as u32) % self.vocab.len() as u32;
            ids.push(id);
        }
        ids
    }

    /// Decode token IDs into a string
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut s = String::new();
        for &id in ids {
            if let Some(token) = self.vocab.get(id as usize) {
                s.push_str(token);
            }
        }
        s
    }

    /// Get BOS token ID if configured
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_basic() {
        // Will test with actual GGUF
    }
}
