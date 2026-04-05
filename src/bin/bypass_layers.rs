/// Test: embed → final_norm → LM_head (skip all layers)
/// If this matches llama.cpp's logits pattern, the issue is in the layers.
use powerinfer::model::InferenceContext;
use powerinfer::ops;
use powerinfer::runtime::BackendFactory;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf";
    std::env::set_var("POWERINFER_NO_CUDA", "1");
    let backend = BackendFactory::cpu();
    let ctx = InferenceContext::from_gguf(model_path, backend)?;
    let weights = ctx.weights();
    let config = ctx.config();
    let rms_eps = config.rms_epsilon;
    let n_embd = config.embedding_length;

    // Get embedding for token 727
    let embd_t = weights.get("token_embd.weight").unwrap();
    let h = embd_t.embedding_row_to_f32(727)?;

    // Apply final norm directly to embedding (no layers)
    let output_norm_w = weights.get_data("output_norm.weight")?;
    let mut normed = vec![0.0f32; n_embd];
    ops::rms_norm(&mut normed, &h, &output_norm_w, rms_eps);

    let norm_val: f32 = normed.iter().map(|x| x * x).sum::<f32>().sqrt();
    eprintln!("Normed embedding: norm={norm_val:.4}");
    eprintln!("First 8: {:?}", &normed[..8]);

    // LM head: output.weight [n_embd, n_vocab]
    let output_t = weights.get("output.weight").unwrap();
    let n_vocab = output_t.shape[1];
    let mut logits = vec![0.0f32; n_vocab];
    powerinfer::quant::matvec_col_major(
        &mut logits,
        &normed,
        output_t.raw(),
        output_t.qtype,
        n_embd,
        n_vocab,
    )?;

    eprintln!("Logits: n_vocab={n_vocab}");

    // Print first 20 logits
    println!("DIRECT_LOGITS_START");
    for (i, logit) in logits.iter().enumerate().take(20) {
        println!("{i} {logit:.6}");
    }

    // Top 10
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("TOP10_START");
    for (idx, val) in indexed.iter().take(10) {
        println!("{idx} {val:.6}");
    }

    // Also test: what if we swap token_embd and output weights?
    // i.e., use token_embd as the LM head weight
    let token_embd_t = weights.get("token_embd.weight").unwrap();
    let mut logits2 = vec![0.0f32; n_vocab];
    powerinfer::quant::matvec_col_major(
        &mut logits2,
        &normed,
        token_embd_t.raw(),
        token_embd_t.qtype,
        n_embd,
        n_vocab,
    )?;

    println!("TOKEN_EMBD_AS_LM_HEAD_TOP10");
    let mut indexed2: Vec<(usize, f32)> =
        logits2.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (idx, val) in indexed2.iter().take(10) {
        println!("{idx} {val:.6}");
    }

    // Check: are output.weight and token_embd.weight the same bytes?
    let out_raw = output_t.raw();
    let emb_raw = token_embd_t.raw();
    let same = out_raw[..100] == emb_raw[..100];
    eprintln!("output.weight == token_embd.weight (first 100 bytes): {same}");

    Ok(())
}
