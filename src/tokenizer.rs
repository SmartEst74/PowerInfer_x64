//! Tokenizer for model vocabulary
//!
//! Loads BPE tokenizer from GGUF metadata. Supports GPT-2, Llama, and Qwen-style
//! tokenizers with byte fallback.

use std::collections::HashMap;

use crate::gguf::GgufFile;
use anyhow::anyhow;

/// Simple BPE tokenizer loaded from GGUF
pub struct Tokenizer {
    /// Token ID → string piece
    vocab: Vec<String>,
    /// Token ID → merge priority score
    scores: Vec<f32>,
    /// String piece → token ID
    token_to_id: HashMap<String, u32>,
    /// BPE merge rules: "token1 token2" → merged_token_id
    merges: HashMap<String, u32>,
    /// Special token IDs
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    #[allow(dead_code)]
    pad_token_id: Option<u32>,
}

impl Tokenizer {
    /// Load tokenizer from GGUF metadata
    pub fn from_gguf(gguf: &GgufFile) -> crate::Result<Self> {
        // Load vocab tokens array
        let tokens_val = gguf
            .metadata("tokenizer.ggml.tokens")
            .ok_or_else(|| anyhow!("Missing tokenizer.ggml.tokens in GGUF metadata"))?;

        let tokens_array = tokens_val
            .as_array()
            .ok_or_else(|| anyhow!("tokenizer.ggml.tokens is not an array"))?;

        let vocab: Vec<String> = tokens_array
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();

        // Load scores array
        let scores_val = gguf.metadata("tokenizer.ggml.scores");
        let scores: Vec<f32> = if let Some(s) = scores_val {
            s.as_array()
                .map(|arr| {
                    arr.iter()
                        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                        .collect()
                })
                .unwrap_or_else(|| vec![0.0; vocab.len()])
        } else {
            vec![0.0; vocab.len()]
        };

        if scores.len() != vocab.len() {
            return Err(anyhow!(
                "vocab/score length mismatch: {} vs {}",
                vocab.len(),
                scores.len()
            ));
        }

        // Build reverse map
        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (i, token) in vocab.iter().enumerate() {
            token_to_id.insert(token.clone(), i as u32);
        }

        // Load merges if present
        let merges = if let Some(merges_val) = gguf.metadata("tokenizer.ggml.merges") {
            if let Some(merges_arr) = merges_val.as_array() {
                let mut map = HashMap::new();
                for (i, merge_val) in merges_arr.iter().enumerate() {
                    if let Some(merge_str) = merge_val.as_str() {
                        // merge format: "token1 token2"
                        map.insert(merge_str.to_string(), (vocab.len() + i) as u32);
                    }
                }
                map
            } else {
                HashMap::new()
            }
        } else {
            HashMap::new()
        };

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
            scores,
            token_to_id,
            merges,
            bos_token_id,
            eos_token_id,
            pad_token_id,
        })
    }

    /// Encode a string into token IDs using BPE with byte fallback
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();

        // Add BOS token if configured
        if let Some(bos) = self.bos_token_id {
            tokens.push(bos);
        }

        // Tokenize word by word, then apply BPE merges
        for word in text.split_whitespace() {
            let word_tokens = self.tokenize_word(word);
            tokens.extend(word_tokens);

            // Add a space token between words (if space token exists)
            if let Some(&space_id) = self.token_to_id.get(" ") {
                tokens.push(space_id);
            }
        }

        // Remove trailing space token
        if let Some(bos) = self.bos_token_id {
            if tokens.last() == self.token_to_id.get(" ").copied().as_ref()
                && tokens.len() > 1
                && tokens[tokens.len() - 2] != bos
            {
                tokens.pop();
            }
        } else if tokens.last() == self.token_to_id.get(" ").copied().as_ref() {
            tokens.pop();
        }

        tokens
    }

    /// Tokenize a single word into initial tokens, then apply BPE merges
    fn tokenize_word(&self, word: &str) -> Vec<u32> {
        // Start with individual characters (or byte tokens)
        let mut tokens: Vec<u32> = Vec::new();
        for ch in word.chars() {
            if let Some(&id) = self.token_to_id.get(&ch.to_string()) {
                tokens.push(id);
            } else {
                // Byte fallback: map each byte to a token
                let mut buf = [0u8; 4];
                let bytes = ch.encode_utf8(&mut buf);
                for &b in bytes.as_bytes() {
                    let byte_token = format!("<0x{b:02X}>");
                    if let Some(&id) = self.token_to_id.get(&byte_token) {
                        tokens.push(id);
                    } else {
                        // Last resort: use a fallback token
                        tokens.push(0);
                    }
                }
            }
        }

        // Apply BPE merges greedily
        self.apply_bpe_merges(&mut tokens);

        tokens
    }

    /// Apply BPE merge rules to a list of token IDs
    fn apply_bpe_merges(&self, tokens: &mut Vec<u32>) {
        loop {
            if tokens.len() < 2 {
                break;
            }

            // Find the best merge (lowest score = highest priority)
            let mut best_pos = None;
            let mut best_score = f32::MAX;

            for i in 0..tokens.len() - 1 {
                let pair = format!(
                    "{} {}",
                    self.safe_token(tokens[i]),
                    self.safe_token(tokens[i + 1])
                );
                if let Some(&_merged_id) = self.merges.get(&pair) {
                    // Use the score of the merged token if available
                    let score = if (_merged_id as usize) < self.scores.len() {
                        self.scores[_merged_id as usize]
                    } else {
                        f32::MAX
                    };
                    if score < best_score {
                        best_score = score;
                        best_pos = Some(i);
                    }
                }
            }

            if let Some(pos) = best_pos {
                let pair = format!(
                    "{} {}",
                    self.safe_token(tokens[pos]),
                    self.safe_token(tokens[pos + 1])
                );
                if let Some(&merged_id) = self.merges.get(&pair) {
                    tokens[pos] = merged_id;
                    tokens.remove(pos + 1);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    /// Get a token string safely
    fn safe_token(&self, id: u32) -> &str {
        self.vocab
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// Decode token IDs into a string
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut s = String::new();
        for &id in ids {
            if let Some(token) = self.vocab.get(id as usize) {
                // Handle byte tokens like <0x20>
                if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
                    if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
                        s.push(byte as char);
                        continue;
                    }
                }
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
    #[test]
    fn test_tokenizer_basic() {
        // Integration test with actual GGUF file
    }
}
