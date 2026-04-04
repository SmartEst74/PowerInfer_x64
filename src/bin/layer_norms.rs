use powerinfer::model::InferenceContext;
use powerinfer::runtime::BackendFactory;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf";
    std::env::set_var("POWERINFER_NO_CUDA", "1");
    std::env::set_var("POWERINFER_TRACE_TOKENS", "1");
    let backend = BackendFactory::cpu();
    let mut ctx = InferenceContext::from_gguf(model_path, backend)?;

    // Process the 19-token fibonacci prompt token by token
    // Only instrument the LAST token (pos=18)
    let prompt = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return";
    let all_ids = ctx.tokenizer().encode(prompt);
    
    // First, process tokens 0-17 WITHOUT diagnostics
    std::env::set_var("POWERINFER_TRACE_TOKENS", "0");
    for &tok in &all_ids[..all_ids.len()-1] {
        ctx.forward(&[tok])?;
    }
    
    // Now process last token WITH diagnostics
    std::env::set_var("POWERINFER_TRACE_TOKENS", "1");
    eprintln!("=== Processing last token (pos=18, token={} = {:?}) ===", 
        all_ids[all_ids.len()-1], ctx.tokenizer().decode(&[all_ids[all_ids.len()-1]]));
    let logits = ctx.forward(&[all_ids[all_ids.len()-1]])?;
    
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for &(id, logit) in indexed.iter().take(5) {
        let text = ctx.tokenizer().decode(&[id as u32]);
        eprintln!("  token={id:>6} logit={logit:>8.4} text={text:?}");
    }
    
    Ok(())
}
