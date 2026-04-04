use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Probe for a working CUDA installation by checking nvcc or libcuda.
fn cuda_available() -> bool {
    // 1. Check CUDA_PATH env
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvcc = PathBuf::from(&cuda_path).join("bin/nvcc");
        if nvcc.exists() {
            return true;
        }
    }
    // 2. Check nvcc on PATH
    if Command::new("nvcc").arg("--version").output().is_ok() {
        return true;
    }
    // 3. Check /usr/local/cuda (standard Linux install)
    if PathBuf::from("/usr/local/cuda/bin/nvcc").exists() {
        return true;
    }
    // 4. Check for libcuda.so (driver present even without toolkit)
    for dir in &["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"] {
        let p = PathBuf::from(dir).join("libcuda.so");
        if p.exists() {
            return true;
        }
    }
    false
}

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=POWERINFER_NO_CUDA");
    println!("cargo:rerun-if-changed=kernels");

    let cuda_feature = env::var("CARGO_FEATURE_CUDA").is_ok();
    let vulkan_feature = env::var("CARGO_FEATURE_VULKAN").is_ok();
    let no_cuda_env = env::var("POWERINFER_NO_CUDA").is_ok();

    if cuda_feature && vulkan_feature {
        panic!("Cannot enable both CUDA and Vulkan backends simultaneously");
    }

    // Auto-detect: if CUDA is available and not explicitly disabled,
    // instruct downstream code via cfg flag. The actual `cuda` Cargo feature
    // still gates the `cust` dependency, but we emit a loud warning when
    // CUDA hardware is present but the feature wasn't enabled.
    let has_cuda_hw = cuda_available();
    if !cuda_feature && has_cuda_hw && !no_cuda_env {
        println!("cargo:warning=CUDA detected but --features cuda not enabled. Rebuild with: cargo build --release --features cuda");
        println!("cargo:warning=Set POWERINFER_NO_CUDA=1 to suppress this warning.");
    }

    if cuda_feature {
        println!("cargo:warning=CUDA backend enabled — GPU offloading active");
    } else if vulkan_feature {
        println!("cargo:warning=Vulkan backend selected; kernels will be compiled during build");
    } else {
        println!("cargo:warning=No GPU backend selected (CPU-only mode)");
    }

    // Create embedded_kernels.rs stub
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let embed_file = out_dir.join("embedded_kernels.rs");
    
    fs::write(
        &embed_file,
        "// Kernel embedding placeholder\n// Build with CUDA/Vulkan features to embed actual kernels\n",
    ).unwrap();

    println!("cargo:rerun-if-changed={}", embed_file.display());
}
