use powerinfer::model::InferenceContext;
use powerinfer::runtime::BackendFactory;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf";
    let backend = BackendFactory::cpu();
    let mut ctx = InferenceContext::from_gguf(model_path, backend)?;

    // Test 1: Raw completion
    println!("=== TEST 1: Raw completion ===");
    test_generate(&mut ctx, "The capital of France is", 10)?;
    ctx.reset();

    // Test 2: Chat template - extended to check for degeneration
    println!("\n=== TEST 2: Chat template (50 tokens) ===");
    test_generate(
        &mut ctx,
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
        50,
    )?;
    ctx.reset();

    // Test 3: Code completion
    println!("\n=== TEST 3: Code completion ===");
    test_generate(
        &mut ctx,
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        15,
    )?;

    Ok(())
}

fn test_generate(
    ctx: &mut InferenceContext,
    prompt: &str,
    n_tokens: usize,
) -> anyhow::Result<()> {
    let input_ids = ctx.tokenizer().encode(prompt);
    println!("Prompt: {:?} ({} tokens)", prompt, input_ids.len());

    let mut logits = ctx.forward(&input_ids)?;

    for step in 0..n_tokens {
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_id = indexed[0].0 as u32;
        let top_text = ctx.tokenizer().decode(&[top_id]);
        let top5: Vec<String> = indexed[..5]
            .iter()
            .map(|(id, l)| {
                format!(
                    "{}({:.1})",
                    ctx.tokenizer().decode(&[*id as u32]).replace('\n', "\\n"),
                    l
                )
            })
            .collect();
        println!(
            "  Step {step}: {:?} | top5: {}",
            top_text,
            top5.join(", ")
        );

        if Some(top_id) == ctx.tokenizer().eos_token_id() {
            break;
        }

        logits = ctx.forward(&[top_id])?;
    }

    Ok(())
}
