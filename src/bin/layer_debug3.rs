use powerinfer::model::InferenceContext;
use powerinfer::runtime::BackendFactory;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf";
    // Force CPU-only to rule out GPU issues
    std::env::set_var("POWERINFER_NO_CUDA", "1");
    let backend = BackendFactory::cpu();
    let mut ctx = InferenceContext::from_gguf(model_path, backend)?;

    let tests: Vec<(&str, &str)> = vec![
        ("1-tok", "The"),
        ("5-tok", "The capital of France is"),
        ("7-tok", "def fibonacci(n):\n    if"),
        ("10-tok", "def fibonacci(n):\n    if n <= 1"),
        (
            "13-tok",
            "def fibonacci(n):\n    if n <= 1:\n        return",
        ),
        (
            "19-tok",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        ),
    ];

    for (label, prompt) in &tests {
        ctx.reset();
        let input_ids = ctx.tokenizer().encode(prompt);
        eprintln!("\n=== {label} ({} tokens) ===", input_ids.len());
        let logits = ctx.forward(&input_ids)?;
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for &(id, logit) in indexed.iter().take(5) {
            let text = ctx.tokenizer().decode(&[id as u32]);
            eprintln!("  token={id:>6} logit={logit:>8.4} text={text:?}");
        }
    }

    Ok(())
}
