/// Dump first 200 logits + spot checks for token 760 ("The")
use powerinfer::model::InferenceContext;
use powerinfer::runtime::BackendFactory;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf";
    std::env::set_var("POWERINFER_NO_CUDA", "1");
    let backend = BackendFactory::cpu();
    let mut ctx = InferenceContext::from_gguf(model_path, backend)?;

    // Run a single token through our forward pass
    let logits = ctx.forward(&[760])?;

    let n_vocab = logits.len();
    println!("n_vocab={n_vocab}");

    println!("LOGITS_START");
    for (i, logit) in logits.iter().enumerate() {
        println!("{i} {logit:.6}");
    }

    let test_indices: Vec<usize> = vec![
        727, 4211, 73111, 164042, 1000, 5000, 10000, 50000, 100000, 200000,
    ];
    println!("SPOT_START");
    for &idx in &test_indices {
        if idx < n_vocab {
            println!("{idx} {:.6}", logits[idx]);
        }
    }

    // Top 10
    println!("TOP10_START");
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (idx, val) in indexed.iter().take(10) {
        println!("{idx} {val:.6}");
    }

    Ok(())
}
