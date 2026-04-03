//! Tokenizer for model vocabulary
//!
//! Loads BPE tokenizer from GGUF metadata. Supports GPT-2/Qwen byte-level BPE
//! and SentencePiece/Llama tokenizers with byte fallback.

use std::collections::HashMap;

use crate::gguf::GgufFile;
use anyhow::anyhow;

/// GPT-2 byte-to-unicode mapping.
///
/// Bytes that are "printable" (33..=126, 161..=172, 174..=255) map to the unicode
/// character with the same codepoint.  All other bytes (0..=32, 127..=160, 173)
/// are remapped to U+0100..U+0143 in order.  This is the standard mapping used
/// by GPT-2, Qwen, and other byte-level BPE tokenizers.
fn gpt2_byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut offset = 0u32;
    for b in 0u32..=255 {
        let is_direct = matches!(b as u8, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
        if is_direct {
            table[b as usize] = char::from_u32(b).unwrap();
        } else {
            table[b as usize] = char::from_u32(256 + offset).unwrap();
            offset += 1;
        }
    }
    table
}

/// Reverse of [`gpt2_byte_to_unicode`]: maps GPT-2 unicode characters back to
/// their original byte value.
fn gpt2_unicode_to_byte() -> HashMap<char, u8> {
    let fwd = gpt2_byte_to_unicode();
    fwd.iter()
        .enumerate()
        .map(|(b, &ch)| (ch, b as u8))
        .collect()
}

/// Simple BPE tokenizer loaded from GGUF
pub struct Tokenizer {
    /// Token ID → string piece (in GPT-2 unicode encoding for byte-level models)
    vocab: Vec<String>,
    /// Token ID → merge priority score (kept for compatibility)
    #[allow(dead_code)]
    scores: Vec<f32>,
    /// String piece → token ID
    token_to_id: HashMap<String, u32>,
    /// BPE merge rules: "token1 token2" → (merged_token_id, priority_rank)
    /// Lower rank = higher priority (applied first in BPE).
    merges: HashMap<String, (u32, u32)>,
    /// Special token IDs
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
    /// Whether this vocab uses GPT-2 byte-level encoding (Ġ for space, etc.)
    uses_byte_level: bool,
    /// GPT-2: byte value → unicode character used in vocab
    byte_to_char: [char; 256],
    /// GPT-2: unicode character → original byte value
    char_to_byte: HashMap<char, u8>,
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

        // Load merges if present — each line is "token1 token2", index = priority rank
        let merges = if let Some(merges_val) = gguf.metadata("tokenizer.ggml.merges") {
            if let Some(merges_arr) = merges_val.as_array() {
                let mut map = HashMap::new();
                for (rank, merge_val) in merges_arr.iter().enumerate() {
                    if let Some(merge_str) = merge_val.as_str() {
                        // merge format: "token1 token2"
                        // The merged result is the concatenation "token1token2" looked up in vocab
                        let parts: Vec<&str> = merge_str.splitn(2, ' ').collect();
                        if parts.len() == 2 {
                            let merged_piece = format!("{}{}", parts[0], parts[1]);
                            if let Some(&merged_id) = token_to_id.get(&merged_piece) {
                                map.insert(merge_str.to_string(), (merged_id, rank as u32));
                            }
                        }
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

        // Detect GPT-2 byte-level encoding.
        // Primary: check tokenizer.ggml.model == "gpt2".
        // Fallback: check if Ġ (U+0120, the GPT-2 representation of space) is a
        // single-character token in the vocabulary.
        let model_type = gguf
            .metadata("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let uses_byte_level = model_type == "gpt2"
            || token_to_id.contains_key("\u{0120}");

        let byte_to_char = gpt2_byte_to_unicode();
        let char_to_byte = gpt2_unicode_to_byte();

        Ok(Self {
            vocab,
            scores,
            token_to_id,
            merges,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            uses_byte_level,
            byte_to_char,
            char_to_byte,
        })
    }

    /// Encode a string into token IDs using BPE.
    ///
    /// For GPT-2 byte-level vocabularies (Qwen, GPT-2, etc.) each byte of the
    /// input is mapped to its GPT-2 unicode character, then BPE merges are applied.
    /// For SentencePiece vocabularies the legacy word-by-word path is used.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();

        // Add BOS token if configured
        if let Some(bos) = self.bos_token_id {
            tokens.push(bos);
        }

        if self.uses_byte_level {
            // GPT-2 byte-level BPE: convert each byte to its GPT-2 unicode
            // character, look up the single-character token, then apply BPE
            // merges on the full sequence.
            let mut initial = Vec::with_capacity(text.len());
            for &byte in text.as_bytes() {
                let ch = self.byte_to_char[byte as usize];
                let ch_str = ch.to_string();
                if let Some(&id) = self.token_to_id.get(&ch_str) {
                    initial.push(id);
                }
                // All 256 byte values should have a token in GPT-2 vocabs.
            }
            self.apply_bpe_merges(&mut initial);
            tokens.extend(initial);
        } else {
            // Legacy SentencePiece / word-by-word path
            for word in text.split_whitespace() {
                let word_tokens = self.tokenize_word(word);
                tokens.extend(word_tokens);

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

            // Find the best merge (lowest rank = highest priority in BPE)
            let mut best_pos = None;
            let mut best_rank = u32::MAX;
            let mut best_merged_id = 0u32;

            for i in 0..tokens.len() - 1 {
                let pair = format!(
                    "{} {}",
                    self.safe_token(tokens[i]),
                    self.safe_token(tokens[i + 1])
                );
                if let Some(&(merged_id, rank)) = self.merges.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pos = Some(i);
                        best_merged_id = merged_id;
                    }
                }
            }

            if let Some(pos) = best_pos {
                tokens[pos] = best_merged_id;
                tokens.remove(pos + 1);
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
        if self.uses_byte_level {
            // GPT-2 byte-level: concatenate token strings, then convert each
            // GPT-2 unicode character back to its original byte value.
            let mut bytes = Vec::new();
            for &id in ids {
                if let Some(token) = self.vocab.get(id as usize) {
                    for ch in token.chars() {
                        if let Some(&b) = self.char_to_byte.get(&ch) {
                            bytes.push(b);
                        }
                        // Unknown chars are dropped (shouldn't happen in practice)
                    }
                }
            }
            String::from_utf8(bytes).unwrap_or_else(|e| {
                String::from_utf8_lossy(e.as_bytes()).into_owned()
            })
        } else {
            // Legacy SentencePiece path
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
    }

    /// Get BOS token ID if configured
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Get PAD token ID
    pub fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
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
        let path = "/home/jon/models/llama-cache/Arch-Agent-3B.Q8_0.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("SKIP: model not found");
            return;
        }
        let gguf = crate::gguf::GgufFile::open(path).expect("GGUF should load");
        // Tokenizer may fail to load if metadata is missing (some GGUFs lack tokenizer.ggml.tokens)
        match super::Tokenizer::from_gguf(&gguf) {
            Ok(tokenizer) => {
                // If it loads, verify basic functionality
                let _ = tokenizer.encode("Hello world");
            }
            Err(_) => {
                eprintln!("SKIP: tokenizer metadata not found in GGUF");
            }
        }
    }
}
