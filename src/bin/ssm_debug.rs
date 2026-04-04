use powerinfer::model::InferenceContext;
use powerinfer::runtime::BackendFactory;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf";
    std::env::set_var("POWERINFER_NO_CUDA", "1");
    let backend = BackendFactory::cpu();
    let mut ctx = InferenceContext::from_gguf(model_path, backend)?;

    // Test: check SSM state norms after various numbers of tokens
    let prompt = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return";
    let all_ids = ctx.tokenizer().encode(prompt);

    // Process token by token, printing SSM state norms at key points
    for (i, &tok) in all_ids.iter().enumerate() {
        ctx.forward(&[tok])?;

        // Print SSM state norms at key positions
        if i == 13 || i == 17 || i == 18 || i == all_ids.len() - 1 {
            let norms = ctx.ssm_state_norms();
            let total_norm: f32 = norms.iter().map(|(_, n)| n * n).sum::<f32>().sqrt();
            let max_layer = norms.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            eprintln!("After pos={i} ({:?}): total_ssm_norm={total_norm:.2} max_layer={:?}",
                ctx.tokenizer().decode(&[tok]), max_layer);
        }
    }

    // Now let's also check: if we process the EXACT same token at different positions
    ctx.reset();

    // Process 14 tokens, then check prediction
    for &tok in &all_ids[..14] {
        ctx.forward(&[tok])?;
    }
    // At position 14, the token is "return" (671)
    let logits_14 = ctx.forward(&[all_ids[14]])?; // token 671 = "return"
    let mut indexed: Vec<(usize, f32)> = logits_14.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("\nPos 14 (first 'return'), top5:");
    for &(id, logit) in indexed.iter().take(5) {
        eprintln!("  {:>6} {:.4} {:?}", id, logit, ctx.tokenizer().decode(&[id as u32]));
    }

    // Continue processing to position 17
    for &tok in &all_ids[15..18] {
        ctx.forward(&[tok])?;
    }
    // At position 18, the token is also "return" (671)
    let logits_18 = ctx.forward(&[all_ids[18]])?; // token 671 = "return"
    let mut indexed: Vec<(usize, f32)> = logits_18.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("\nPos 18 (second 'return'), top5:");
    for &(id, logit) in indexed.iter().take(5) {
        eprintln!("  {:>6} {:.4} {:?}", id, logit, ctx.tokenizer().decode(&[id as u32]));
    }

    // Compare logit vectors
    let cos_sim = |a: &[f32], b: &[f32]| -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb + 1e-12)
    };
    let sim = cos_sim(&logits_14, &logits_18);
    eprintln!("\nCosine sim between pos-14 and pos-18 logits: {sim:.4}");

    Ok(())
}
