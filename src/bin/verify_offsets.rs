use powerinfer::gguf::GgufFile;
use powerinfer::weights::Weights;
use std::io::{Read, Seek};

fn main() -> anyhow::Result<()> {
    let model_path = "/home/jon/models/llama-cache/Qwen3.5-35B-A3B-Q8_0.gguf";

    // Open GGUF and get tensor info from gguf_rs
    let gguf = GgufFile::open(model_path)?;

    println!("=== Tensor info from gguf_rs ===");
    for t in gguf.tensors() {
        if t.name == "token_embd.weight"
            || t.name == "blk.0.attn_norm.weight"
            || t.name == "output.weight"
        {
            println!(
                "  {} offset={} size={} shape={:?} kind={:?}",
                t.name, t.offset, t.size, t.shape, t.kind
            );
        }
    }

    // Parse header manually
    let mut f = std::fs::File::open(model_path)?;
    let mut b4 = [0u8; 4];
    let mut b8 = [0u8; 8];

    // Read magic
    f.read_exact(&mut b4)?;
    println!("\n=== GGUF Header ===");
    println!("  magic: {:?}", std::str::from_utf8(&b4));

    // Read version
    f.read_exact(&mut b4)?;
    let version = u32::from_le_bytes(b4);
    println!("  version: {version}");

    // Read tensor count
    f.read_exact(&mut b8)?;
    let nt = u64::from_le_bytes(b8);
    println!("  tensor_count: {nt}");

    // Read metadata count
    f.read_exact(&mut b8)?;
    let nm = u64::from_le_bytes(b8);
    println!("  metadata_count: {nm}");
    println!("  current position after header: {}", f.stream_position()?);

    // Now let's check what our parse_header computes
    // We'll call it via Weights
    std::env::set_var("POWERINFER_NO_CUDA", "1");
    let weights = Weights::from_gguf(&gguf)?;

    let embd = weights.get("token_embd.weight").unwrap();
    let norm0 = weights.get("blk.0.attn_norm.weight").unwrap();

    println!("\n=== From our Weights (shape+qtype) ===");
    println!(
        "  token_embd.weight: shape={:?} qtype={:?}",
        embd.shape, embd.qtype
    );
    println!(
        "  blk.0.attn_norm.weight: shape={:?} qtype={:?}",
        norm0.shape, norm0.qtype
    );

    // Read the first 16 raw bytes of token_embd
    let embd_raw = embd.raw();
    println!("\n=== First 16 raw bytes of token_embd.weight ===");
    print!("  mmap: ");
    for b in &embd_raw[..16] {
        print!("{b:02x} ");
    }
    println!();
    println!("  total raw bytes: {}", embd_raw.len());

    // Verify blk.0.attn_norm.weight (should be F32)
    let norm0_raw = norm0.raw();
    let first4: [u8; 4] = norm0_raw[..4].try_into().unwrap();
    let first_f32 = f32::from_le_bytes(first4);
    println!("\n=== blk.0.attn_norm.weight first value (F32) ===");
    println!(
        "  bytes: {:02x} {:02x} {:02x} {:02x}",
        first4[0], first4[1], first4[2], first4[3]
    );
    println!("  f32: {first_f32}");

    // Dump a few more norm values
    print!("  first 8 values:");
    for i in 0..8 {
        let off = i * 4;
        let b: [u8; 4] = norm0_raw[off..off + 4].try_into().unwrap();
        let v = f32::from_le_bytes(b);
        print!(" {v:.6}");
    }
    println!();

    Ok(())
}
