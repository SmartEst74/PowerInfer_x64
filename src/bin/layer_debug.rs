use powerinfer::model::InferenceContext;
use powerinfer::runtime::BackendFactory;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf";
    let backend = BackendFactory::cpu();
    let mut ctx = InferenceContext::from_gguf(model_path, backend)?;

    // Use the fibonacci prompt (the one that breaks)
    let prompt = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return";
    let input_ids = ctx.tokenizer().encode(prompt);
    eprintln!("Prompt tokens ({}):", input_ids.len());
    for &id in &input_ids {
        eprintln!("  {} -> {:?}", id, ctx.tokenizer().decode(&[id]));
    }

    // Enable diagnostics
    std::env::set_var("POWERINFER_DIAG", "1");

    // Run prefill
    eprintln!("\n=== PREFILL ({} tokens) ===", input_ids.len());
    let logits = ctx.forward(&input_ids)?;

    // Print top-10 logits
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("\n=== TOP 10 LOGITS (after prefill) ===");
    for (rank, &(id, logit)) in indexed.iter().take(10).enumerate() {
        let text = ctx.tokenizer().decode(&[id as u32]);
        eprintln!("  #{rank}: token={id} logit={logit:.4} text={text:?}");
    }

    // Also print logit stats
    let n_nan = logits.iter().filter(|x| x.is_nan()).count();
    let n_inf = logits.iter().filter(|x| x.is_infinite()).count();
    let mean: f32 = logits.iter().sum::<f32>() / logits.len() as f32;
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    eprintln!("\nLogit stats: len={}, nan={n_nan}, inf={n_inf}, min={min:.4}, max={max:.4}, mean={mean:.6}", logits.len());

    // Now decode one token from prefill
    let next_id = indexed[0].0 as u32;
    let next_text = ctx.tokenizer().decode(&[next_id]);
    eprintln!("\nFirst generated token: {} -> {:?}", next_id, next_text);

    // Decode a few more tokens
    for step in 0..5 {
        let logits = ctx.forward(&[next_id])?;
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let next_id_new = indexed[0].0 as u32;
        let text = ctx.tokenizer().decode(&[next_id_new]);
        eprintln!(
            "Step {step}: token={next_id_new} text={text:?} logit={:.4}",
            indexed[0].1
        );
    }

    Ok(())
}
