use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-changed=kernels");

    let features = env::var("CARGO_FEATURE_CUDA").is_ok();
    let vulkan_features = env::var("CARGO_FEATURE_VULKAN").is_ok();

    if features && vulkan_features {
        panic!("Cannot enable both CUDA and Vulkan backends simultaneously");
    }

    // For now, just inform about selected backend
    // Full kernel compilation will be implemented when kernels are ready
    if features {
        println!("cargo:warning=CUDA backend selected; kernels will be compiled during build");
        // Placeholder: we could copy precompiled PTX here
    } else if vulkan_features {
        println!("cargo:warning=Vulkan backend selected; kernels will be compiled during build");
    } else {
        println!("cargo:warning=No GPU backend selected (CPU-only mode)");
    }

    // Create embedded_kernels.rs stub
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let embed_file = out_dir.join("embedded_kernels.rs");
    
    // For now, just embed an empty placeholder
    // Real implementation will compile kernels from kernels/*.rs
    fs::write(
        &embed_file,
        "// Kernel embedding placeholder\n// Build with CUDA/Vulkan features to embed actual kernels\n",
    ).unwrap();

    println!("cargo:rerun-if-changed={}", embed_file.display());
}
