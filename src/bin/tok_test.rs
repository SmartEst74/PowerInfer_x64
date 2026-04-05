use powerinfer::model::InferenceContext;
use powerinfer::runtime::BackendFactory;

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf".to_string());
    std::env::set_var("POWERINFER_NO_CUDA", "1");
    let backend = BackendFactory::cpu();
    let ctx = InferenceContext::from_gguf(&model_path, backend)?;
    let tok = ctx.tokenizer();

    let tests: &[&str] = &[
        "The capital of France is",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "/no_think",
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41",
        " 31, 37, 41, 43, 47",
        "41",
        " 41",
        " ZZ",
    ];

    for s in tests {
        let ids = tok.encode(s);
        let back = tok.decode(&ids);
        println!("{s:50} -> ids={ids:?}  back={back:?}");
    }

    println!("\nToken IDs 40..46 decode as:");
    for id in 40u32..46 {
        let decoded = tok.decode(&[id]);
        println!("  {id} -> {decoded:?}");
    }
    Ok(())
}
