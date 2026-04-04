/// Layer-by-layer debug: dump hidden state at each step for token 727
use powerinfer::model::InferenceContext;
use powerinfer::runtime::BackendFactory;
use powerinfer::ops;

fn norm_and_stats(label: &str, v: &[f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  {label}: len={} norm={norm:.6} min={min:.6} max={max:.6}", v.len());
    print!("    first 8:");
    for x in v.iter().take(8) { print!(" {x:.6}"); }
    println!();
}

fn main() -> anyhow::Result<()> {
    let model_path = "/home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf";
    std::env::set_var("POWERINFER_NO_CUDA", "1");
    let backend = BackendFactory::cpu();
    let ctx = InferenceContext::from_gguf(model_path, backend)?;
    let weights = ctx.weights();
    let config = ctx.config();
    let rms_eps = config.rms_epsilon;
    let n_embd = config.embedding_length;

    // Step 1: Get embedding for token 727
    let embd_t = weights.get("token_embd.weight").unwrap();
    let h = embd_t.embedding_row_to_f32(727)?;
    println!("=== Token 727 embedding ===");
    norm_and_stats("embed", &h);

    // Step 2: Layer 0 attn_norm (RMSNorm)
    let attn_norm_w = weights.get_data("blk.0.attn_norm.weight")?;
    let mut normed = vec![0.0f32; n_embd];
    ops::rms_norm(&mut normed, &h, &attn_norm_w, rms_eps);
    println!("\n=== Layer 0: after RMSNorm ===");
    norm_and_stats("normed", &normed);

    // Step 3: QKV projection
    let inproj_t = weights.get("blk.0.attn_qkv.weight").unwrap();
    let conv_dim = inproj_t.shape.get(1).copied().unwrap_or(0);
    let mut qkv = vec![0.0f32; conv_dim];
    powerinfer::quant::matvec_col_major(
        &mut qkv, &normed, inproj_t.raw(), inproj_t.qtype, n_embd, conv_dim,
    )?;
    println!("\n=== Layer 0: after QKV projection (before conv) ===");
    norm_and_stats("qkv", &qkv);
    
    // Step 4: Conv1d (first token — conv state is zeros except last slot)
    let ssm_conv1d_w = weights.get_data("blk.0.ssm_conv1d.weight")?;
    let d_conv = 4;
    let mut qkv_conv = vec![0.0f32; conv_dim];
    for i in 0..conv_dim {
        // First token: only kernel_idx=3 (last conv weight) applies
        let w_idx = i * d_conv + (d_conv - 1);
        let val = ssm_conv1d_w[w_idx] * qkv[i];
        // SiLU
        qkv_conv[i] = val / (1.0 + (-val).exp());
    }
    println!("\n=== Layer 0: after Conv1d+SiLU ===");
    norm_and_stats("qkv_conv", &qkv_conv);
    
    // Step 5: Split Q, K, V
    let ssm_out_t = weights.get("blk.0.ssm_out.weight").unwrap();
    let value_dim = ssm_out_t.shape.first().copied().unwrap_or(0);
    let key_dim = (conv_dim - value_dim) / 2;
    println!("\n  key_dim={key_dim} value_dim={value_dim} conv_dim={conv_dim}");
    
    let q_raw = &qkv_conv[..key_dim];
    let k_raw = &qkv_conv[key_dim..key_dim * 2];
    let v_raw = &qkv_conv[key_dim * 2..];
    norm_and_stats("Q", q_raw);
    norm_and_stats("K", k_raw);
    norm_and_stats("V", v_raw);

    // Step 6: Z gate
    let gate_t = weights.get("blk.0.attn_gate.weight").unwrap();
    let mut z = vec![0.0f32; value_dim];
    powerinfer::quant::matvec_col_major(
        &mut z, &normed, gate_t.raw(), gate_t.qtype, n_embd, value_dim,
    )?;
    norm_and_stats("Z gate", &z);

    // Step 7: Beta, Alpha
    let ssm_beta_t = weights.get("blk.0.ssm_beta.weight").unwrap();
    let ssm_alpha_t = weights.get("blk.0.ssm_alpha.weight").unwrap();
    let ssm_norm_w = weights.get_data("blk.0.ssm_norm.weight")?;
    let v_hd = ssm_norm_w.len();
    let n_v_h = if v_hd > 0 { value_dim / v_hd } else { 1 };
    let k_hd = if n_v_h > 0 { key_dim / (n_v_h / 2).max(1) } else { 1 };
    let n_k_h = if k_hd > 0 { key_dim / k_hd } else { 1 };
    println!("\n  v_hd={v_hd} n_v_h={n_v_h} k_hd={k_hd} n_k_h={n_k_h}");

    let mut beta_raw = vec![0.0f32; n_v_h];
    powerinfer::quant::matvec_col_major(
        &mut beta_raw, &normed, ssm_beta_t.raw(), ssm_beta_t.qtype, n_embd, n_v_h,
    )?;
    let mut beta = beta_raw.clone();
    for b in beta.iter_mut() { *b = 1.0 / (1.0 + (-*b).exp()); }
    println!("\n");
    norm_and_stats("beta (sigmoid)", &beta);

    let mut a_raw = vec![0.0f32; n_v_h];
    powerinfer::quant::matvec_col_major(
        &mut a_raw, &normed, ssm_alpha_t.raw(), ssm_alpha_t.qtype, n_embd, n_v_h,
    )?;
    let ssm_a_data = weights.get_data("blk.0.ssm_a")?;
    let ssm_dt_bias = weights.get_data("blk.0.ssm_dt.bias")?;
    let mut g_decay = vec![0.0f32; n_v_h];
    for h in 0..n_v_h {
        let ssm_a = ssm_a_data[h];
        let sp_in = a_raw[h] + ssm_dt_bias[h];
        let sp = if sp_in > 10.0 { sp_in } else { (1.0_f32 + sp_in.exp()).ln() };
        g_decay[h] = ssm_a * sp;
    }
    norm_and_stats("g_decay", &g_decay);

    // Step 8: Repeat-interleave Q, K
    let mut q_exp = vec![0.0f32; n_v_h * k_hd];
    let mut k_exp = vec![0.0f32; n_v_h * k_hd];
    for vh in 0..n_v_h {
        let kh = vh % n_k_h;
        let src = kh * k_hd;
        let dst = vh * k_hd;
        q_exp[dst..dst + k_hd].copy_from_slice(&q_raw[src..src + k_hd]);
        k_exp[dst..dst + k_hd].copy_from_slice(&k_raw[src..src + k_hd]);
    }
    
    // L2 normalize
    let q_scale = 1.0 / (k_hd as f32).sqrt();
    for vh in 0..n_v_h {
        let off = vh * k_hd;
        let q_slice = &mut q_exp[off..off + k_hd];
        let norm_q: f32 = q_slice.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in q_slice.iter_mut() { *x = *x / norm_q * q_scale; }
        let k_slice = &mut k_exp[off..off + k_hd];
        let norm_k: f32 = k_slice.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in k_slice.iter_mut() { *x /= norm_k; }
    }
    
    // Step 9: Delta rule (state starts at 0)
    // For first token with zero state: out[v] = beta * (Q·K) * V[v]
    let mut attn_result = vec![0.0f32; value_dim];
    for vh in 0..n_v_h {
        let beta_h = beta[vh];
        let q_h = &q_exp[vh * k_hd..(vh + 1) * k_hd];
        let k_h = &k_exp[vh * k_hd..(vh + 1) * k_hd];
        let v_h = &v_raw[vh * v_hd..(vh + 1) * v_hd];
        
        // State = 0, so:
        // S[k,v] = K[k] * V[v] * beta
        // out[v] = sum_k S[k,v] * Q[k] = beta * V[v] * (Q·K)
        let qk_dot: f32 = q_h.iter().zip(k_h).map(|(q, k)| q * k).sum();
        for (vi, &v_val) in v_h.iter().enumerate() {
            attn_result[vh * v_hd + vi] = beta_h * qk_dot * v_val;
        }
    }
    println!("\n=== Layer 0: after delta rule ===");
    norm_and_stats("attn_result", &attn_result);

    // Step 10: RMSNormGated
    for vh in 0..n_v_h {
        let off = vh * v_hd;
        let out_h = &mut attn_result[off..off + v_hd];
        let z_h = &z[off..off + v_hd];
        let var: f32 = out_h.iter().map(|x| x * x).sum::<f32>() / v_hd as f32;
        let scale = 1.0 / (var + rms_eps).sqrt();
        for (i, o) in out_h.iter_mut().enumerate() {
            let normed_val = *o * scale * ssm_norm_w[i];
            let z_val = z_h[i];
            let silu_z = z_val / (1.0 + (-z_val).exp());
            *o = normed_val * silu_z;
        }
    }
    println!("\n=== Layer 0: after RMSNormGated ===");
    norm_and_stats("gated_output", &attn_result);

    // Step 11: Output projection
    let mut out = vec![0.0f32; n_embd];
    powerinfer::quant::matvec_col_major(
        &mut out, &attn_result, ssm_out_t.raw(), ssm_out_t.qtype, value_dim, n_embd,
    )?;
    println!("\n=== Layer 0: output projection ===");
    norm_and_stats("out_proj", &out);

    // Step 12: Residual
    let mut h_out = h.clone();
    for (ho, o) in h_out.iter_mut().zip(out.iter()) { *ho += o; }
    println!("\n=== Layer 0: after residual (h + attn_out) ===");
    norm_and_stats("h_resid", &h_out);

    Ok(())
}
