use powerinfer::model::InferenceContext;
use powerinfer::runtime::BackendFactory;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf";
    std::env::set_var("POWERINFER_NO_CUDA", "1");
    let backend = BackendFactory::cpu();
    let mut ctx = InferenceContext::from_gguf(model_path, backend)?;

    // Test 1: Single token "def" (727)
    let tokens1: Vec<u32> = vec![727];
    let logits1 = ctx.forward(&tokens1)?;

    println!("=== Single token test (token=727 'def') ===");
    println!("n_vocab: {}", logits1.len());

    let mut indexed: Vec<(usize, f32)> = logits1.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top-5 logits:");
    for &(id, logit) in indexed.iter().take(5) {
        let text = ctx.tokenizer().decode(&[id as u32]);
        println!("  token={id} logit={logit:.6} text={text:?}");
    }

    // Specific logits for comparison
    println!("\nSpecific logit values:");
    let check_tokens = [
        (73111, " fibonacci"),
        (727, "def"),
        (760, "The"),
        (220, " "),
        (198, "\\n"),
        (164042, "teria"),
    ];
    for &(tok, name) in &check_tokens {
        if tok < logits1.len() {
            println!("  token={tok} ({name}): {:.6}", logits1[tok]);
        }
    }

    // Test 2: Full fibonacci (19 tokens)
    ctx.reset();
    let tokens2: Vec<u32> = vec![
        727, 73111, 1393, 1590, 198, 262, 413, 307, 2564, 220, 16, 25, 198, 285, 460, 307, 198,
        262, 460,
    ];
    let logits2 = ctx.forward(&tokens2)?;

    println!("\n=== Full fibonacci (19 tokens) ===");

    let mut indexed2: Vec<(usize, f32)> = logits2.iter().copied().enumerate().collect();
    indexed2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top-5 logits:");
    for &(id, logit) in indexed2.iter().take(5) {
        let text = ctx.tokenizer().decode(&[id as u32]);
        println!("  token={id} logit={logit:.6} text={text:?}");
    }

    println!("\nSpecific logit values:");
    for &(tok, name) in &check_tokens {
        if tok < logits2.len() {
            println!("  token={tok} ({name}): {:.6}", logits2[tok]);
        }
    }

    Ok(())
}
