use powerinfer::model::InferenceContext;
use powerinfer::runtime::BackendFactory;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf";
    let backend = BackendFactory::cpu();
    let mut ctx = InferenceContext::from_gguf(model_path, backend)?;

    // Test 1: Short prompt (works)
    let prompt = "The capital of France is";
    let input_ids = ctx.tokenizer().encode(prompt);
    eprintln!("=== SHORT PROMPT ({} tokens) ===", input_ids.len());
    let logits = ctx.forward(&input_ids)?;
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (rank, &(id, logit)) in indexed.iter().take(5).enumerate() {
        let text = ctx.tokenizer().decode(&[id as u32]);
        eprintln!("  #{rank}: token={id} logit={logit:.4} text={text:?}");
    }

    ctx.reset();

    // Test 2: 1-token prompt
    let prompt = "The";
    let input_ids = ctx.tokenizer().encode(prompt);
    eprintln!("\n=== SINGLE TOKEN ({} tokens) ===", input_ids.len());
    let logits = ctx.forward(&input_ids)?;
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (rank, &(id, logit)) in indexed.iter().take(5).enumerate() {
        let text = ctx.tokenizer().decode(&[id as u32]);
        eprintln!("  #{rank}: token={id} logit={logit:.4} text={text:?}");
    }

    ctx.reset();

    // Test 3: Medium prompt - just 3 tokens
    let prompt = "def fibonacci";
    let input_ids = ctx.tokenizer().encode(prompt);
    eprintln!("\n=== MEDIUM PROMPT ({} tokens) ===", input_ids.len());
    let logits = ctx.forward(&input_ids)?;
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (rank, &(id, logit)) in indexed.iter().take(5).enumerate() {
        let text = ctx.tokenizer().decode(&[id as u32]);
        eprintln!("  #{rank}: token={id} logit={logit:.4} text={text:?}");
    }

    ctx.reset();

    // Test 4: 10 tokens
    let prompt = "def fibonacci(n):\n    if";
    let input_ids = ctx.tokenizer().encode(prompt);
    eprintln!("\n=== 10-TOKEN PROMPT ({} tokens) ===", input_ids.len());
    let logits = ctx.forward(&input_ids)?;
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (rank, &(id, logit)) in indexed.iter().take(5).enumerate() {
        let text = ctx.tokenizer().decode(&[id as u32]);
        eprintln!("  #{rank}: token={id} logit={logit:.4} text={text:?}");
    }

    Ok(())
}
