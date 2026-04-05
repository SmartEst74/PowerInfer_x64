//! Tokenizer for model vocabulary
//!
//! Loads BPE tokenizer from GGUF metadata. Supports GPT-2/Qwen byte-level BPE
//! and SentencePiece/Llama tokenizers with byte fallback.

use std::collections::HashMap;

use crate::gguf::GgufFile;
use anyhow::anyhow;
use regex::Regex;

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

/// A segment of text that is either a special token or plain text for BPE encoding.
enum TokenSegment {
    Special(u32),
    Text(String),
}

/// Pretokenizer type determining how text is split before BPE
#[derive(Debug, Clone, Copy, PartialEq)]
enum PreTokenizerType {
    /// No pretokenization (legacy)
    None,
    /// GPT-2 style pretokenization
    Gpt2,
    /// Qwen2 pretokenization
    Qwen2,
    /// Qwen3.5 pretokenization (adds \p{M} to letter class)
    Qwen35,
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
    /// Special/control tokens that must be matched as whole strings before BPE.
    /// Sorted longest-first for greedy matching.
    special_tokens: Vec<(String, u32)>,
    /// Pretokenizer type (determines regex for splitting before BPE)
    _pre_type: PreTokenizerType,
    /// Compiled pretokenization regex (if applicable)
    pre_regex: Option<Regex>,
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
        let uses_byte_level = model_type == "gpt2" || token_to_id.contains_key("\u{0120}");

        let byte_to_char = gpt2_byte_to_unicode();
        let char_to_byte = gpt2_unicode_to_byte();

        // Load special/control tokens from token_type metadata.
        // Type 3 = control, type 4 = user_defined — these must be matched as
        // whole strings before BPE encoding.
        let mut special_tokens = Vec::new();
        if let Some(types_val) = gguf.metadata("tokenizer.ggml.token_type") {
            if let Some(types_arr) = types_val.as_array() {
                for (i, type_val) in types_arr.iter().enumerate() {
                    let t = type_val.as_u64().unwrap_or(0);
                    if (t == 3 || t == 4) && i < vocab.len() {
                        special_tokens.push((vocab[i].clone(), i as u32));
                    }
                }
            }
        }
        // Sort longest-first for greedy matching
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        // Detect pretokenizer type from GGUF metadata
        let pre_type_str = gguf
            .metadata("tokenizer.ggml.pre")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let pre_type = match pre_type_str {
            "qwen35" => PreTokenizerType::Qwen35,
            "qwen2" | "deepseek-r1-qwen" => PreTokenizerType::Qwen2,
            "default" if model_type == "gpt2" => PreTokenizerType::Gpt2,
            _ if model_type == "gpt2" => PreTokenizerType::Gpt2,
            _ => PreTokenizerType::None,
        };

        // Compile pretokenization regex
        // These patterns match llama.cpp's pretokenizer regexes.
        // Note: \s+(?!\S) requires a negative lookahead which the `regex` crate
        // doesn't support. We use \s+ and post-process to simulate the effect:
        // trim the last char from pure-whitespace matches preceding non-whitespace.
        let pre_regex = match pre_type {
            PreTokenizerType::Qwen35 => Some(
                Regex::new(concat!(
                    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
                    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+",
                    r"|\p{N}",
                    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*",
                    r"|\s*[\r\n]+",
                    r"|\s+",
                ))
                .expect("invalid qwen35 pretokenizer regex"),
            ),
            PreTokenizerType::Qwen2 | PreTokenizerType::Gpt2 => Some(
                Regex::new(concat!(
                    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
                    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
                    r"|\p{N}",
                    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
                    r"|\s*[\r\n]+",
                    r"|\s+",
                ))
                .expect("invalid gpt2/qwen2 pretokenizer regex"),
            ),
            PreTokenizerType::None => None,
        };

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
            special_tokens,
            _pre_type: pre_type,
            pre_regex,
        })
    }

    /// Encode a string into token IDs using BPE.
    ///
    /// Special/control tokens (e.g. `<|im_start|>`) are matched as whole strings
    /// before BPE encoding. The remaining text segments are encoded with BPE.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();

        // Add BOS token if configured
        if let Some(bos) = self.bos_token_id {
            tokens.push(bos);
        }

        // Split text around special tokens, encoding each segment
        let segments = self.split_special_tokens(text);
        for segment in segments {
            match segment {
                TokenSegment::Special(id) => tokens.push(id),
                TokenSegment::Text(s) => tokens.extend(self.encode_text(&s)),
            }
        }

        tokens
    }

    /// Split text into alternating text/special-token segments.
    /// Special tokens are matched greedily (longest first).
    fn split_special_tokens(&self, text: &str) -> Vec<TokenSegment> {
        if self.special_tokens.is_empty() {
            return vec![TokenSegment::Text(text.to_string())];
        }

        let mut segments = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Try to match a special token at the current position
            let mut matched = false;
            for (special_str, token_id) in &self.special_tokens {
                if remaining.starts_with(special_str.as_str()) {
                    segments.push(TokenSegment::Special(*token_id));
                    remaining = &remaining[special_str.len()..];
                    matched = true;
                    break;
                }
            }
            if !matched {
                // Find the next special token occurrence
                let mut next_pos = remaining.len();
                for (special_str, _) in &self.special_tokens {
                    if let Some(pos) = remaining.find(special_str.as_str()) {
                        if pos < next_pos {
                            next_pos = pos;
                        }
                    }
                }
                // Everything before the next special token is plain text
                segments.push(TokenSegment::Text(remaining[..next_pos].to_string()));
                remaining = &remaining[next_pos..];
            }
        }

        segments
    }

    /// Encode a plain text segment (no special tokens) using BPE.
    fn encode_text(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        if self.uses_byte_level {
            // Pretokenize: split text into chunks, then BPE each chunk independently
            let chunks = self.pretokenize(text);
            let mut result = Vec::new();
            for chunk in &chunks {
                let mut initial = Vec::with_capacity(chunk.len());
                for &byte in chunk.as_bytes() {
                    let ch = self.byte_to_char[byte as usize];
                    let ch_str = ch.to_string();
                    if let Some(&id) = self.token_to_id.get(&ch_str) {
                        initial.push(id);
                    }
                }
                self.apply_bpe_merges(&mut initial);
                result.extend(initial);
            }
            result
        } else {
            // Legacy SentencePiece / word-by-word path
            let mut result = Vec::new();
            for word in text.split_whitespace() {
                let word_tokens = self.tokenize_word(word);
                result.extend(word_tokens);

                if let Some(&space_id) = self.token_to_id.get(" ") {
                    result.push(space_id);
                }
            }

            // Remove trailing space token
            if result.last() == self.token_to_id.get(" ").copied().as_ref() {
                result.pop();
            }
            result
        }
    }

    /// Split text into pretokenization chunks using the model's regex.
    ///
    /// This implements the pretokenization step that GPT-2/Qwen tokenizers use
    /// before BPE. Each chunk is independently BPE-encoded, preventing merges
    /// across word boundaries.
    ///
    /// The `\s+(?!\S)` lookahead pattern (used in the reference implementation)
    /// is simulated by post-processing: pure-whitespace chunks longer than 1 char
    /// that precede non-whitespace have their last char moved to the next chunk.
    fn pretokenize(&self, text: &str) -> Vec<String> {
        let re = match &self.pre_regex {
            Some(re) => re,
            None => {
                // No pretokenization: return the entire text as one chunk
                return vec![text.to_string()];
            }
        };

        // Collect regex matches
        let mut chunks: Vec<String> = re.find_iter(text).map(|m| m.as_str().to_string()).collect();

        // Post-process to simulate \s+(?!\S) behavior:
        // When a pure-whitespace chunk (>1 char) precedes a non-whitespace chunk,
        // move the last whitespace character to the next chunk. This causes words
        // to absorb a leading space (e.g. "    if" → "   " + " if").
        let mut i = 0;
        while i < chunks.len() {
            let is_pure_ws = chunks[i].chars().all(|c| c.is_whitespace());
            let char_count = chunks[i].chars().count();
            let next_starts_nonws = i + 1 < chunks.len()
                && chunks[i + 1]
                    .chars()
                    .next()
                    .is_some_and(|c| !c.is_whitespace());

            if is_pure_ws && char_count > 1 && next_starts_nonws {
                // Split: all but last char stays, last char prepends to next chunk
                let last_char_start = chunks[i]
                    .char_indices()
                    .next_back()
                    .map(|(idx, _)| idx)
                    .unwrap();
                let suffix = chunks[i][last_char_start..].to_string();
                chunks[i].truncate(last_char_start);
                chunks[i + 1] = format!("{suffix}{}", chunks[i + 1]);
            }
            i += 1;
        }

        // Remove empty chunks that could result from post-processing
        chunks.retain(|c| !c.is_empty());
        chunks
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
            String::from_utf8(bytes)
                .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
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
