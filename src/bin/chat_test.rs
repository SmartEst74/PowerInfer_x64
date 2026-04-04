use powerinfer::model::InferenceContext;
use powerinfer::runtime::BackendFactory;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf";
    std::env::set_var("POWERINFER_NO_CUDA", "1");
    let backend = BackendFactory::cpu();
    let mut ctx = InferenceContext::from_gguf(model_path, backend)?;

    // Test 1: Raw fibonacci prompt (known broken at 19 tokens)
    let raw = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return";
    let raw_ids = ctx.tokenizer().encode(raw);
    eprintln!("=== RAW ({} tokens) ===", raw_ids.len());
    let logits = ctx.forward(&raw_ids)?;
    print_top(&ctx, &logits, 5);

    ctx.reset();

    // Test 2: Chat template wrapped
    let chat = "<|im_start|>user\nComplete this function:\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return<|im_end|>\n<|im_start|>assistant\n";
    let chat_ids = ctx.tokenizer().encode(chat);
    eprintln!("\n=== CHAT ({} tokens) ===", chat_ids.len());
    eprintln!("Token IDs: {:?}", &chat_ids[..10.min(chat_ids.len())]);
    let logits = ctx.forward(&chat_ids)?;
    print_top(&ctx, &logits, 5);

    ctx.reset();

    // Test 3: Just the raw "return" with more context
    let raw2 = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1)";
    let raw2_ids = ctx.tokenizer().encode(raw2);
    eprintln!("\n=== LONGER RAW ({} tokens) ===", raw2_ids.len());
    let logits = ctx.forward(&raw2_ids)?;
    print_top(&ctx, &logits, 5);

    ctx.reset();

    // Test 4: What about our token-by-token generation from position 14?
    // Process 15 tokens (positions 0-14), then generate 5 tokens
    for &tok in &raw_ids[..15] {
        ctx.forward(&[tok])?;
    }
    eprintln!("\n=== TOKEN-BY-TOKEN from pos 15 ===");
    // Process remaining 4 tokens
    for (i, &tok) in raw_ids[15..].iter().enumerate() {
        let logits = ctx.forward(&[tok])?;
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_text = ctx.tokenizer().decode(&[indexed[0].0 as u32]);
        eprintln!("  pos={}: input={:?} top={:?}({:.2})", 15+i, ctx.tokenizer().decode(&[tok]), top_text, indexed[0].1);
    }

    Ok(())
}

fn print_top(ctx: &InferenceContext, logits: &[f32], n: usize) {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for &(id, logit) in indexed.iter().take(n) {
        let text = ctx.tokenizer().decode(&[id as u32]);
        eprintln!("  {:>6} {:.4} {:?}", id, logit, text);
    }
}
