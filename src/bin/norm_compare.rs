use powerinfer::model::InferenceContext;
use powerinfer::runtime::BackendFactory;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf";
    std::env::set_var("POWERINFER_NO_CUDA", "1");
    let backend = BackendFactory::cpu();
    let mut ctx = InferenceContext::from_gguf(model_path, backend)?;

    // Test 1: Working prompt - process token by token, show norms for LAST token
    let prompt1 = "The capital of France is";
    let ids1 = ctx.tokenizer().encode(prompt1);
    for &tok in &ids1[..ids1.len() - 1] {
        ctx.forward(&[tok])?;
    }
    std::env::set_var("POWERINFER_TRACE_TOKENS", "1");
    eprintln!(
        "=== GOOD prompt: last tok = {} ({:?}) ===",
        ids1[ids1.len() - 1],
        ctx.tokenizer().decode(&[ids1[ids1.len() - 1]])
    );
    let logits = ctx.forward(&[ids1[ids1.len() - 1]])?;
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!(
        "Top: {:?}({:.2})",
        ctx.tokenizer().decode(&[indexed[0].0 as u32]),
        indexed[0].1
    );

    ctx.reset();
    std::env::set_var("POWERINFER_TRACE_TOKENS", "0");

    // Test 2: Broken prompt
    let prompt2 = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return";
    let ids2 = ctx.tokenizer().encode(prompt2);
    for &tok in &ids2[..ids2.len() - 1] {
        ctx.forward(&[tok])?;
    }
    std::env::set_var("POWERINFER_TRACE_TOKENS", "1");
    eprintln!(
        "\n=== BAD prompt: last tok = {} ({:?}) ===",
        ids2[ids2.len() - 1],
        ctx.tokenizer().decode(&[ids2[ids2.len() - 1]])
    );
    let logits = ctx.forward(&[ids2[ids2.len() - 1]])?;
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!(
        "Top: {:?}({:.2})",
        ctx.tokenizer().decode(&[indexed[0].0 as u32]),
        indexed[0].1
    );

    ctx.reset();
    std::env::set_var("POWERINFER_TRACE_TOKENS", "0");

    // Test 3: Same length (19 tokens) but working English prompt
    let prompt3 = "The quick brown fox jumps over the lazy dog and then the cat sat on a beautiful";
    let ids3 = ctx.tokenizer().encode(prompt3);
    // Trim/pad to exactly 19
    let ids3 = &ids3[..ids3.len().min(19)];
    for &tok in &ids3[..ids3.len() - 1] {
        ctx.forward(&[tok])?;
    }
    std::env::set_var("POWERINFER_TRACE_TOKENS", "1");
    eprintln!(
        "\n=== ENGLISH 19tok: last tok = {} ({:?}) ===",
        ids3[ids3.len() - 1],
        ctx.tokenizer().decode(&[ids3[ids3.len() - 1]])
    );
    let logits = ctx.forward(&[ids3[ids3.len() - 1]])?;
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!(
        "Top: {:?}({:.2})",
        ctx.tokenizer().decode(&[indexed[0].0 as u32]),
        indexed[0].1
    );

    Ok(())
}
