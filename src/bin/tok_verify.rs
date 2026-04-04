use powerinfer::model::InferenceContext;
use powerinfer::runtime::BackendFactory;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf";
    std::env::set_var("POWERINFER_NO_CUDA", "1");
    let backend = BackendFactory::cpu();
    let mut ctx = InferenceContext::from_gguf(model_path, backend)?;

    // llama.cpp's tokenization of the fibonacci prompt
    let llamacpp_tokens: Vec<u32> = vec![727, 73111, 1393, 1590, 198, 262, 413, 307, 2564, 220, 16, 25, 198, 285, 460, 307, 198, 262, 460];
    // Our tokenization
    let our_tokens: Vec<u32> = vec![727, 73111, 1393, 1590, 198, 257, 331, 307, 2564, 220, 16, 25, 198, 260, 671, 307, 198, 257, 671];

    // Show what each token decodes to
    eprintln!("=== Token comparison ===");
    for i in 0..llamacpp_tokens.len() {
        let lc = llamacpp_tokens[i];
        let ours = our_tokens[i];
        let lc_text = ctx.tokenizer().decode(&[lc]);
        let our_text = ctx.tokenizer().decode(&[ours]);
        let marker = if lc != ours { "  ← DIFF" } else { "" };
        eprintln!("  pos={i:>2}: llama={lc:>6} ({lc_text:>12?})  ours={ours:>6} ({our_text:>12?}){marker}");
    }

    // Test 1: Run with llama.cpp tokens
    eprintln!("\n=== Using llama.cpp tokens ===");
    let logits = ctx.forward(&llamacpp_tokens)?;
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for &(id, logit) in indexed.iter().take(5) {
        let text = ctx.tokenizer().decode(&[id as u32]);
        eprintln!("  {:>6} {:.4} {:?}", id, logit, text);
    }

    ctx.reset();

    // Test 2: Run with our tokens
    eprintln!("\n=== Using our tokens ===");
    let logits = ctx.forward(&our_tokens)?;
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for &(id, logit) in indexed.iter().take(5) {
        let text = ctx.tokenizer().decode(&[id as u32]);
        eprintln!("  {:>6} {:.4} {:?}", id, logit, text);
    }

    Ok(())
}
