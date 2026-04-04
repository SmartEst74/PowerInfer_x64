use powerinfer::model::InferenceContext;
use powerinfer::runtime::BackendFactory;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf";
    std::env::set_var("POWERINFER_NO_CUDA", "1");
    let backend = BackendFactory::cpu();
    let mut ctx = InferenceContext::from_gguf(model_path, backend)?;

    // Test the fibonacci prompt at different lengths
    let prompt = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return";
    let all_ids = ctx.tokenizer().encode(prompt);
    eprintln!("Full prompt: {} tokens = {:?}", all_ids.len(), all_ids);

    // Test each prefix length
    for n in [15, 16, 17, 18, 19] {
        if n > all_ids.len() { break; }
        ctx.reset();
        let ids = &all_ids[..n];
        let logits = ctx.forward(ids)?;
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top = &indexed[..3];
        let top_str: Vec<String> = top.iter().map(|&(id, l)| {
            format!("{}({:.2})", ctx.tokenizer().decode(&[id as u32]).replace('\n', "\\n"), l)
        }).collect();
        eprintln!("{n:>2} tokens: {}", top_str.join(", "));
    }

    // Now test a DIFFERENT 19+ token English prompt
    ctx.reset();
    let eng_prompt = "The quick brown fox jumps over the lazy dog and the cat sat on the mat near the";
    let eng_ids = ctx.tokenizer().encode(eng_prompt);
    eprintln!("\nEnglish test ({} tokens):", eng_ids.len());
    let logits = ctx.forward(&eng_ids)?;
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for &(id, logit) in indexed.iter().take(5) {
        let text = ctx.tokenizer().decode(&[id as u32]);
        eprintln!("  token={id:>6} logit={logit:>8.4} text={text:?}");
    }

    // Also test: repeat a single sentence to make a long prompt
    ctx.reset();
    let repeat_prompt = "I like cats. I like cats. I like cats. I like cats. I like";
    let rep_ids = ctx.tokenizer().encode(repeat_prompt);
    eprintln!("\nRepeat test ({} tokens):", rep_ids.len());
    let logits = ctx.forward(&rep_ids)?;
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for &(id, logit) in indexed.iter().take(5) {
        let text = ctx.tokenizer().decode(&[id as u32]);
        eprintln!("  token={id:>6} logit={logit:>8.4} text={text:?}");
    }

    // Also test: process the 19-token fibonacci prompt token by token
    ctx.reset();
    eprintln!("\n=== TOKEN-BY-TOKEN fibonacci (19 tok) ===");
    for (i, &tok) in all_ids.iter().enumerate() {
        let logits = ctx.forward(&[tok])?;
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_id = indexed[0].0 as u32;
        let top_tok = ctx.tokenizer().decode(&[top_id]);
        let expected = if i + 1 < all_ids.len() {
            ctx.tokenizer().decode(&[all_ids[i + 1]])
        } else {
            "<END>".to_string()
        };
        eprintln!("  After {} (pos={i}): top={top_tok:?}({:.2}) expect={expected:?}", 
            ctx.tokenizer().decode(&[tok]),
            indexed[0].1);
    }

    Ok(())
}
