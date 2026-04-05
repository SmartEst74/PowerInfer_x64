use powerinfer::model::InferenceContext;
use powerinfer::runtime::BackendFactory;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf";
    std::env::set_var("POWERINFER_NO_CUDA", "1");
    let backend = BackendFactory::cpu();
    let ctx = InferenceContext::from_gguf(model_path, backend)?;

    let embd_tensor = ctx.weights().get("token_embd.weight").unwrap();

    println!("tensor name: {}", embd_tensor.name);
    println!("shape: {:?}", embd_tensor.shape);
    println!("qtype: {:?}", embd_tensor.qtype);
    println!("raw bytes: {}", embd_tensor.raw().len());

    // For token 727, check first Q8_0 block manually
    let ne0 = embd_tensor.shape[0]; // 2048
    let bs = 34; // Q8_0 block size
    let vp = 32; // values per block
    let bytes_per_row = (ne0 / vp) * bs; // 64 * 34 = 2176
    let start = 727 * bytes_per_row;

    let raw = embd_tensor.raw();
    println!("\nToken 727 embedding start offset: {start}");
    println!("Bytes per embedding: {bytes_per_row}");

    // First Q8_0 block: 2 bytes scale (f16) + 32 bytes weights
    let scale_bytes = [raw[start], raw[start + 1]];
    let scale = half::f16::from_le_bytes(scale_bytes).to_f32();
    println!("\nFirst Q8_0 block:");
    println!(
        "  scale bytes: [{:#04x}, {:#04x}]",
        raw[start],
        raw[start + 1]
    );
    println!("  scale (f16->f32): {scale}");

    // First 16 int8 weights
    print!("  int8 weights:");
    for i in 0..16 {
        let q = raw[start + 2 + i] as i8;
        print!(" {q}");
    }
    println!();

    // Dequantized first 16 values
    print!("  dequantized:");
    for i in 0..16 {
        let q = raw[start + 2 + i] as i8 as f32;
        print!(" {:.6}", q * scale);
    }
    println!();

    // Compare with embedding_row_to_f32
    let emb = embd_tensor.embedding_row_to_f32(727)?;
    println!("\nembedding_row_to_f32 first 16: {:?}", &emb[..16]);

    // Also check token 0 to see if embeddings look reasonable
    let emb0 = embd_tensor.embedding_row_to_f32(0)?;
    let norm0: f32 = emb0.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("\nToken 0 embedding norm: {norm0:.6}");
    println!("Token 0 first 4: {:?}", &emb0[..4]);

    // Check: what does llama.cpp read for the same block?
    // The raw offset in the GGUF file should be data_section_start + tensor_offset + 727 * bytes_per_row
    // Let me also print the total tensor byte_len for verification
    println!("\nTotal tensor byte_len: {}", raw.len());
    println!(
        "Expected: {} * {} = {}",
        embd_tensor.shape[1],
        bytes_per_row,
        embd_tensor.shape[1] as u64 * bytes_per_row as u64
    );

    Ok(())
}
