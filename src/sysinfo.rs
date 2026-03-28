//! System resource detector and GPU memory planner
//!
//! Scans available hardware (GPU VRAM, CPU cores, RAM) and plans
//! how to distribute model weights across GPU and CPU for optimal
//! inference performance.

use std::process::Command;

/// Detected GPU device
#[derive(Debug, Clone)]
pub struct GpuDevice {
    /// Device index (0, 1, ...)
    pub index: usize,
    /// GPU name (e.g., "NVIDIA GeForce GTX 1050 Ti")
    pub name: String,
    /// Total VRAM in bytes
    pub total_vram: u64,
    /// Free VRAM in bytes
    pub free_vram: u64,
    /// Used VRAM in bytes
    pub used_vram: u64,
    /// CUDA compute capability major version
    pub compute_major: u32,
    /// CUDA compute capability minor version
    pub compute_minor: u32,
}

/// System resource summary
#[derive(Debug, Clone)]
pub struct SystemResources {
    /// Available GPU devices
    pub gpus: Vec<GpuDevice>,
    /// Total system RAM in bytes
    pub total_ram: u64,
    /// Available system RAM in bytes
    pub available_ram: u64,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// CUDA version string
    pub cuda_version: Option<String>,
}

/// GPU memory allocation plan
#[derive(Debug, Clone)]
pub struct GpuMemoryPlan {
    /// Per-GPU allocations
    pub gpu_allocations: Vec<GpuAllocation>,
    /// Total GPU memory to use
    pub total_gpu_memory: u64,
    /// Total CPU memory to use
    pub total_cpu_memory: u64,
}

/// Allocation for a single GPU
#[derive(Debug, Clone)]
pub struct GpuAllocation {
    /// GPU index
    pub gpu_index: usize,
    /// Weight memory in bytes
    pub weight_bytes: u64,
    /// KV cache memory in bytes
    pub kv_cache_bytes: u64,
    /// Total memory to allocate
    pub total_bytes: u64,
    /// Which layers go on this GPU (if layer-based offloading)
    pub layer_range: Option<(usize, usize)>,
    /// Which neuron indices go on this GPU (if neuron-level sparsity)
    pub hot_neurons_per_layer: Option<Vec<usize>>,
}

impl SystemResources {
    /// Scan the system for available resources
    pub fn scan() -> Self {
        let gpus = detect_gpus();
        let (total_ram, available_ram) = detect_ram();
        let cpu_cores = num_cpus::get();
        let cuda_version = detect_cuda_version();

        Self {
            gpus,
            total_ram,
            available_ram,
            cpu_cores,
            cuda_version,
        }
    }

    /// Total available GPU VRAM across all devices
    pub fn total_gpu_vram(&self) -> u64 {
        self.gpus.iter().map(|g| g.free_vram).sum()
    }

    /// Print a summary of detected resources
    pub fn print_summary(&self) {
        println!("=== System Resources ===");
        println!(
            "  RAM: {:.1} GB total, {:.1} GB available",
            self.total_ram as f64 / (1024.0 * 1024.0 * 1024.0),
            self.available_ram as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!("  CPU: {} cores", self.cpu_cores);
        if let Some(ref cuda) = self.cuda_version {
            println!("  CUDA: {cuda}");
        }
        println!("  GPUs: {}", self.gpus.len());
        for gpu in &self.gpus {
            println!(
                "    [{}] {} — {:.1} GB VRAM ({:.1} GB free), compute {}.{}",
                gpu.index,
                gpu.name,
                gpu.total_vram as f64 / (1024.0 * 1024.0 * 1024.0),
                gpu.free_vram as f64 / (1024.0 * 1024.0 * 1024.0),
                gpu.compute_major,
                gpu.compute_minor,
            );
        }
    }

    /// Plan how to distribute a model across available GPUs
    ///
    /// - `model_bytes`: total model weight size in bytes
    /// - `kv_cache_per_token_bytes`: KV cache size per token in bytes
    /// - `target_context`: target context length in tokens
    pub fn plan_gpu_allocation(
        &self,
        model_bytes: u64,
        kv_cache_per_token_bytes: u64,
        target_context: usize,
    ) -> GpuMemoryPlan {
        let total_vram: u64 = self.gpus.iter().map(|g| g.free_vram).sum();
        let kv_cache_bytes = kv_cache_per_token_bytes * target_context as u64;

        // Reserve 512MB per GPU for CUDA overhead
        let cuda_overhead = self.gpus.len() as u64 * 512 * 1024 * 1024;
        let usable_vram = total_vram.saturating_sub(cuda_overhead);

        if model_bytes + kv_cache_bytes <= usable_vram {
            // Everything fits on GPU
            self.plan_full_gpu(model_bytes, kv_cache_bytes)
        } else if model_bytes <= usable_vram {
            // Model fits but full KV cache doesn't — use compressed KV
            let max_kv = usable_vram - model_bytes;
            let max_context = (max_kv / kv_cache_per_token_bytes) as usize;
            println!(
                "  Note: Full {target_context}-token KV cache doesn't fit. Max context: {max_context}"
            );
            self.plan_full_gpu(model_bytes, max_kv)
        } else {
            // Model is too large for GPU — split across GPU and CPU
            self.plan_split_gpu_cpu(model_bytes, kv_cache_bytes, usable_vram)
        }
    }

    fn plan_full_gpu(&self, model_bytes: u64, kv_cache_bytes: u64) -> GpuMemoryPlan {
        let mut allocations = Vec::new();
        let mut remaining_model = model_bytes;
        let mut remaining_kv = kv_cache_bytes;

        for gpu in &self.gpus {
            let available = gpu.free_vram - 512 * 1024 * 1024; // 512MB overhead
            let weight = remaining_model.min(available / 2);
            let kv = remaining_kv.min(available - weight);

            allocations.push(GpuAllocation {
                gpu_index: gpu.index,
                weight_bytes: weight,
                kv_cache_bytes: kv,
                total_bytes: weight + kv,
                layer_range: None,
                hot_neurons_per_layer: None,
            });

            remaining_model -= weight;
            remaining_kv -= kv;
        }

        GpuMemoryPlan {
            gpu_allocations: allocations,
            total_gpu_memory: model_bytes + kv_cache_bytes,
            total_cpu_memory: 0,
        }
    }

    fn plan_split_gpu_cpu(
        &self,
        model_bytes: u64,
        kv_cache_bytes: u64,
        usable_vram: u64,
    ) -> GpuMemoryPlan {
        // Put as much as possible on GPU, rest on CPU
        let gpu_model_bytes = usable_vram;
        let cpu_model_bytes = model_bytes - gpu_model_bytes;

        let allocations = vec![GpuAllocation {
            gpu_index: 0,
            weight_bytes: gpu_model_bytes,
            kv_cache_bytes: 0,
            total_bytes: gpu_model_bytes,
            layer_range: None,
            hot_neurons_per_layer: None,
        }];

        println!(
            "  Split: {:.1} GB on GPU, {:.1} GB on CPU",
            gpu_model_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            cpu_model_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        GpuMemoryPlan {
            gpu_allocations: allocations,
            total_gpu_memory: gpu_model_bytes,
            total_cpu_memory: cpu_model_bytes + kv_cache_bytes,
        }
    }
}

/// Detect NVIDIA GPUs using nvidia-smi
fn detect_gpus() -> Vec<GpuDevice> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.free,memory.used,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            stdout
                .lines()
                .filter_map(|line| {
                    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if parts.len() < 6 {
                        return None;
                    }
                    let compute: Vec<&str> = parts[5].split('.').collect();
                    Some(GpuDevice {
                        index: parts[0].parse().ok()?,
                        name: parts[1].to_string(),
                        total_vram: parts[2].parse::<u64>().ok()? * 1024 * 1024,
                        free_vram: parts[3].parse::<u64>().ok()? * 1024 * 1024,
                        used_vram: parts[4].parse::<u64>().ok()? * 1024 * 1024,
                        compute_major: compute.first().and_then(|s| s.parse().ok()).unwrap_or(0),
                        compute_minor: compute.get(1).and_then(|s| s.parse().ok()).unwrap_or(0),
                    })
                })
                .collect()
        }
        _ => Vec::new(),
    }
}

/// Detect system RAM from /proc/meminfo
fn detect_ram() -> (u64, u64) {
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let mut total = 0u64;
    let mut available = 0u64;

    for line in meminfo.lines() {
        if line.starts_with("MemTotal:") {
            total = parse_meminfo_kb(line) * 1024;
        } else if line.starts_with("MemAvailable:") {
            available = parse_meminfo_kb(line) * 1024;
        }
    }

    (total, available)
}

fn parse_meminfo_kb(line: &str) -> u64 {
    line.split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Detect CUDA version from nvidia-smi
fn detect_cuda_version() -> Option<String> {
    Command::new("nvidia-smi")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| {
            let stdout = String::from_utf8_lossy(&out.stdout);
            stdout
                .lines()
                .find(|l| l.contains("CUDA Version"))
                .and_then(|l| {
                    l.split("CUDA Version: ")
                        .nth(1)
                        .map(|v| v.trim().to_string())
                })
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_scan() {
        let resources = SystemResources::scan();
        resources.print_summary();
        // On this system, we expect 2 GPUs with 4GB each
        assert!(!resources.gpus.is_empty(), "No GPUs detected");
    }

    #[test]
    fn test_memory_plan_qwen3_4b() {
        let resources = SystemResources::scan();

        // Qwen3-4B Q4_K_M: ~2.4GB weights
        let model_bytes = 2_400_000_000u64;
        // KV cache per token: ~80 bytes * 8 KV heads * 36 layers * 128 head_dim
        let kv_per_token = 80 * 8 * 36 * 128;

        let plan = resources.plan_gpu_allocation(model_bytes, kv_per_token, 4096);

        println!("Memory plan:");
        for alloc in &plan.gpu_allocations {
            println!(
                "  GPU {}: {:.1} GB weights, {:.1} GB KV cache",
                alloc.gpu_index,
                alloc.weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                alloc.kv_cache_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
        println!(
            "  CPU: {:.1} GB",
            plan.total_cpu_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        // With 8GB VRAM, model should fit entirely on GPU
        assert!(
            plan.total_gpu_memory >= model_bytes,
            "Model should fit on GPU"
        );
    }
}
