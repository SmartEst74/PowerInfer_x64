use powerinfer::model::InferenceContext;
use powerinfer::runtime::BackendFactory;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf";
    std::env::set_var("POWERINFER_NO_CUDA", "1");
    let backend = BackendFactory::cpu();
    let ctx = InferenceContext::from_gguf(model_path, backend)?;

    // Dump embedding for token 727 ("def")
    let embd_tensor = ctx.weights().get("token_embd.weight").unwrap();
    let emb = embd_tensor.embedding_row_to_f32(727)?;
    
    println!("Embedding for token 727 ('def'), n_embd={}", emb.len());
    println!("First 16 values: {:?}", &emb[..16]);
    println!("Last 4 values: {:?}", &emb[emb.len()-4..]);
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    let sum: f32 = emb.iter().sum();
    println!("norm={norm:.6}, sum={sum:.6}");
    
    // Also dump output.weight row for token 73111 (" fibonacci")  
    let output_tensor = ctx.weights().get("output.weight").unwrap();
    let out_row = output_tensor.embedding_row_to_f32(73111)?;
    println!("\nOutput weight for token 73111 (' fibonacci'), len={}", out_row.len());
    println!("First 16 values: {:?}", &out_row[..16]);
    let out_norm: f32 = out_row.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("norm={out_norm:.6}");

    // Dump output_norm.weight
    let norm_w = ctx.weights().get_data("output_norm.weight")?;
    println!("\noutput_norm.weight, len={}", norm_w.len());
    println!("First 16 values: {:?}", &norm_w[..16]);
    let nw_norm: f32 = norm_w.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("norm={nw_norm:.6}");

    Ok(())
}
