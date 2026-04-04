use powerinfer::model::InferenceContext;
use powerinfer::runtime::BackendFactory;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf";
    let backend = BackendFactory::cpu();
    let ctx = InferenceContext::from_gguf(model_path, backend)?;
    
    let prompts = vec![
        "The capital of France is",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    ];
    
    for p in &prompts {
        let ids = ctx.tokenizer().encode(p);
        println!("Prompt: {:?}", p);
        println!("  Tokens ({}): {:?}", ids.len(), ids);
        for &id in &ids {
            println!("    {} -> {:?}", id, ctx.tokenizer().decode(&[id]));
        }
    }
    
    Ok(())
}
