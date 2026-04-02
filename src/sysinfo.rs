//! System resource detector, hardware profiler, and execution planner.
//!
//! Performs a complete sweep of CPU capabilities, GPU devices, memory,
//! PCIe topology, and storage I/O to build a `HardwareProfile`.
//! The `ExecutionPlan` then maps model layers to devices for optimal throughput.

use std::collections::HashMap;
use std::process::Command;

// ---------------------------------------------------------------------------
// Memory guard — detect commit-limit headroom before loading a large model
// ---------------------------------------------------------------------------

/// Snapshot of /proc/meminfo commit-accounting fields.
#[derive(Debug, Clone)]
pub struct MemoryGuard {
    /// CommitLimit from /proc/meminfo (kB)
    pub commit_limit_kb: u64,
    /// Committed_AS from /proc/meminfo (kB)
    pub committed_as_kb: u64,
    /// MemAvailable from /proc/meminfo (kB)
    pub mem_available_kb: u64,
    /// SwapFree from /proc/meminfo (kB)
    pub swap_free_kb: u64,
    /// SwapTotal from /proc/meminfo (kB)
    pub swap_total_kb: u64,
    /// Current overcommit_memory setting (0=heuristic, 1=always, 2=never)
    pub overcommit_mode: u8,
}

impl MemoryGuard {
    /// Read current memory accounting from /proc/meminfo.
    pub fn check() -> Self {
        let mut commit_limit_kb = 0u64;
        let mut committed_as_kb = 0u64;
        let mut mem_available_kb = 0u64;
        let mut swap_free_kb = 0u64;
        let mut swap_total_kb = 0u64;

        if let Ok(txt) = std::fs::read_to_string("/proc/meminfo") {
            for line in txt.lines() {
                let mut parts = line.split_whitespace();
                let key = parts.next().unwrap_or("");
                let val: u64 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                match key {
                    "CommitLimit:" => commit_limit_kb = val,
                    "Committed_AS:" => committed_as_kb = val,
                    "MemAvailable:" => mem_available_kb = val,
                    "SwapFree:" => swap_free_kb = val,
                    "SwapTotal:" => swap_total_kb = val,
                    _ => {}
                }
            }
        }

        let overcommit_mode = std::fs::read_to_string("/proc/sys/vm/overcommit_memory")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        Self {
            commit_limit_kb,
            committed_as_kb,
            mem_available_kb,
            swap_free_kb,
            swap_total_kb,
            overcommit_mode,
        }
    }

    /// How many kB remain before the commit limit is hit.
    /// Only meaningful when overcommit_mode == 0 (heuristic).
    pub fn commit_headroom_kb(&self) -> i64 {
        self.commit_limit_kb as i64 - self.committed_as_kb as i64
    }

    /// Print a warning if the machine may OOM during inference of `model_bytes`.
    ///
    /// Returns `true` if the situation looks safe, `false` if risky.
    pub fn preflight(&self, model_bytes: u64) -> bool {
        let model_kb = model_bytes / 1024;
        let headroom_kb = self.commit_headroom_kb();
        let available_gb = self.mem_available_kb as f64 / (1024.0 * 1024.0);
        let model_gb = model_kb as f64 / (1024.0 * 1024.0);
        let headroom_gb = headroom_kb as f64 / (1024.0 * 1024.0);

        eprintln!("[memory] Available RAM  : {available_gb:.1} GB");
        eprintln!("[memory] Model size     : {model_gb:.1} GB (file-backed mmap, not committed)");

        if self.overcommit_mode != 0 {
            // Overcommit disabled — no commit limit enforced
            eprintln!("[memory] Overcommit mode: {} (commit limit not enforced)", self.overcommit_mode);
            return true;
        }

        eprintln!(
            "[memory] Commit headroom: {headroom_gb:.1} GB  (CommitLimit={:.1} GB  Committed_AS={:.1} GB)",
            self.commit_limit_kb as f64 / (1024.0 * 1024.0),
            self.committed_as_kb as f64 / (1024.0 * 1024.0),
        );

        // The inference engine keeps heap allocations below ~1 GB (scratch buffers +
        // one-at-a-time expert matrices ≈ 174 MB peak).  Warn if headroom is tight.
        let safe = headroom_kb >= 1_200_000; // 1.2 GB minimum

        if !safe {
            eprintln!("[memory] WARNING: commit headroom {headroom_gb:.1} GB is dangerously low.");
            eprintln!("[memory]   The OOM killer may terminate this process or the desktop session.");
            eprintln!("[memory]   Recommended fix — run ONE of the following:");
            eprintln!("[memory]");
            eprintln!("[memory]   Option A — disable commit limit (instant, until reboot):");
            eprintln!("[memory]     sudo sysctl -w vm.overcommit_memory=1");
            eprintln!("[memory]");
            eprintln!("[memory]   Option B — add 16 GB swap (survives reboot):");
            eprintln!("[memory]     sudo fallocate -l 16G /swapfile2");
            eprintln!("[memory]     sudo chmod 600 /swapfile2");
            eprintln!("[memory]     sudo mkswap /swapfile2");
            eprintln!("[memory]     sudo swapon /swapfile2");
            eprintln!(
                "[memory]     # This raises CommitLimit to ~{:.0} GB",
                (self.commit_limit_kb as f64 + 8_000_000.0) / (1024.0 * 1024.0)
            );
        }
        safe
    }
}

// ---------------------------------------------------------------------------
// CPU capability detection — SIMD, cores, cache, microarch
// ---------------------------------------------------------------------------

/// CPU SIMD instruction set support
#[derive(Debug, Clone, Default)]
pub struct CpuCapabilities {
    pub model_name: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub sockets: usize,
    pub base_mhz: f64,
    /// Cache sizes in bytes: L1d, L2, L3
    pub cache_l1d: usize,
    pub cache_l2: usize,
    pub cache_l3: usize,
    // SIMD tiers (each implies the previous)
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
    pub sse41: bool,
    pub sse42: bool,
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
    pub avx512f: bool,
    pub avx512vnni: bool,
    // Useful extras
    pub popcnt: bool,
    pub aes: bool,
    pub pclmul: bool,
}

impl CpuCapabilities {
    /// Detect from /proc/cpuinfo and lscpu on Linux.
    pub fn detect() -> Self {
        let mut cap = Self::default();

        // Parse /proc/cpuinfo flags
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in cpuinfo.lines() {
                if line.starts_with("model name") {
                    if cap.model_name.is_empty() {
                        cap.model_name = line.split(':').nth(1).unwrap_or("").trim().to_string();
                    }
                } else if line.starts_with("cpu MHz") {
                    if let Some(mhz) = line.split(':').nth(1).and_then(|s| s.trim().parse::<f64>().ok()) {
                        if mhz > cap.base_mhz { cap.base_mhz = mhz; }
                    }
                } else if line.starts_with("flags") || line.starts_with("Features") {
                    let flags_str = line.split(':').nth(1).unwrap_or("");
                    let flags: Vec<&str> = flags_str.split_whitespace().collect();
                    cap.sse2 = flags.contains(&"sse2");
                    cap.sse3 = flags.contains(&"pni") || flags.contains(&"sse3");
                    cap.ssse3 = flags.contains(&"ssse3");
                    cap.sse41 = flags.contains(&"sse4_1");
                    cap.sse42 = flags.contains(&"sse4_2");
                    cap.avx = flags.contains(&"avx");
                    cap.avx2 = flags.contains(&"avx2");
                    cap.fma = flags.contains(&"fma");
                    cap.f16c = flags.contains(&"f16c");
                    cap.avx512f = flags.contains(&"avx512f");
                    cap.avx512vnni = flags.contains(&"avx512_vnni") || flags.contains(&"avx512vnni");
                    cap.popcnt = flags.contains(&"popcnt");
                    cap.aes = flags.contains(&"aes");
                    cap.pclmul = flags.contains(&"pclmulqdq");
                }
            }
        }

        // Core/thread counts from num_cpus
        cap.logical_cores = num_cpus::get();
        cap.physical_cores = num_cpus::get_physical();
        cap.sockets = 1; // safe default

        // Parse lscpu for cache info
        if let Ok(out) = Command::new("lscpu").arg("-B").output() {
            if out.status.success() {
                let stdout = String::from_utf8_lossy(&out.stdout);
                for line in stdout.lines() {
                    let parts: Vec<&str> = line.splitn(2, ':').collect();
                    if parts.len() < 2 { continue; }
                    let key = parts[0].trim();
                    let val = parts[1].trim();
                    match key {
                        "L1d cache" => cap.cache_l1d = parse_cache_bytes(val),
                        "L2 cache" => cap.cache_l2 = parse_cache_bytes(val),
                        "L3 cache" => cap.cache_l3 = parse_cache_bytes(val),
                        "Socket(s)" => cap.sockets = val.parse().unwrap_or(1),
                        _ => {}
                    }
                }
            }
        }

        cap
    }

    /// The best SIMD tier name this CPU supports.
    pub fn best_simd(&self) -> &'static str {
        if self.avx512vnni { "AVX-512 VNNI" }
        else if self.avx512f { "AVX-512F" }
        else if self.avx2 && self.fma { "AVX2+FMA" }
        else if self.avx2 { "AVX2" }
        else if self.avx { "AVX" }
        else if self.sse42 { "SSE4.2" }
        else if self.sse41 { "SSE4.1" }
        else if self.ssse3 { "SSSE3" }
        else if self.sse2 { "SSE2" }
        else { "Scalar" }
    }

    /// Estimated single-core throughput for Q8_0 matvec (GOPS/s) given SIMD width.
    /// Based on measured benchmarks: SSE4.1 ~2 GOPS, AVX2+FMA ~8 GOPS, AVX-512 ~16 GOPS.
    pub fn estimated_matvec_gops(&self) -> f64 {
        if self.avx512f { 16.0 }
        else if self.avx2 && self.fma { 8.0 }
        else if self.avx2 { 5.0 }
        else if self.sse41 { 2.0 }
        else { 0.8 }
    }
}

fn parse_cache_bytes(s: &str) -> usize {
    // lscpu -B prints bytes like "65536 (2 instances)" — take first token
    s.split_whitespace()
        .next()
        .and_then(|tok| tok.trim().parse::<usize>().ok())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// PCIe topology
// ---------------------------------------------------------------------------

/// PCIe link info for a GPU.
#[derive(Debug, Clone)]
pub struct PcieLink {
    pub gpu_index: usize,
    /// Generation (1, 2, 3, 4, 5)
    pub gen: u32,
    /// Width (x1, x4, x8, x16)
    pub width: u32,
    /// Theoretical unidirectional bandwidth in MB/s
    pub bandwidth_mbs: f64,
}

impl PcieLink {
    /// Detect PCIe link for a GPU device via nvidia-smi.
    pub fn detect(gpu_index: usize) -> Option<Self> {
        let output = Command::new("nvidia-smi")
            .args([
                &format!("--id={gpu_index}"),
                "--query-gpu=pcie.link.gen.current,pcie.link.width.current",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;
        if !output.status.success() { return None; }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = stdout.trim().split(',').map(|s| s.trim()).collect();
        if parts.len() < 2 { return None; }
        let gen: u32 = parts[0].parse().ok()?;
        let width: u32 = parts[1].parse().ok()?;
        // Per-lane bandwidth: Gen1=250 MB/s, Gen2=500, Gen3=985, Gen4=1969, Gen5=3938
        let per_lane = match gen {
            1 => 250.0,
            2 => 500.0,
            3 => 985.0,
            4 => 1969.0,
            5 => 3938.0,
            _ => 250.0,
        };
        Some(Self {
            gpu_index,
            gen,
            width,
            bandwidth_mbs: per_lane * width as f64,
        })
    }
}

// ---------------------------------------------------------------------------
// Storage I/O profiling
// ---------------------------------------------------------------------------

/// Storage device info.
#[derive(Debug, Clone)]
pub struct StorageInfo {
    /// Device name (sda, nvme0n1, etc.)
    pub device: String,
    /// Whether rotational (HDD=true, SSD/NVMe=false).
    pub rotational: bool,
    /// Transport (sata, nvme, usb, etc.)
    pub transport: String,
    /// Measured sequential read MB/s (0 if not measured).
    pub seq_read_mbs: f64,
}

impl StorageInfo {
    /// Detect storage devices from lsblk.
    pub fn detect_all() -> Vec<Self> {
        let output = Command::new("lsblk")
            .args(["-d", "-o", "NAME,ROTA,TYPE,TRAN", "--noheadings"])
            .output();
        match output {
            Ok(out) if out.status.success() => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                stdout.lines().filter_map(|line| {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() < 4 { return None; }
                    if parts[2] != "disk" { return None; }
                    Some(StorageInfo {
                        device: parts[0].to_string(),
                        rotational: parts[1] == "1",
                        transport: parts[3].to_string(),
                        seq_read_mbs: 0.0,
                    })
                }).collect()
            }
            _ => Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Complete hardware profile
// ---------------------------------------------------------------------------

/// Full hardware profile from system sweep.
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub cpu: CpuCapabilities,
    pub memory: MemoryGuard,
    pub gpus: Vec<GpuDevice>,
    pub pcie: Vec<PcieLink>,
    pub storage: Vec<StorageInfo>,
    pub has_igp: bool,
    pub igp_name: Option<String>,
    pub cuda_version: Option<String>,
    /// Total usable VRAM across all GPUs (after 512 MB overhead each).
    pub total_usable_vram: u64,
    /// Total CPU RAM available.
    pub available_ram: u64,
}

impl HardwareProfile {
    /// Perform a complete system sweep.
    pub fn sweep() -> Self {
        let cpu = CpuCapabilities::detect();
        let memory = MemoryGuard::check();
        let gpus = detect_gpus();
        let pcie: Vec<PcieLink> = gpus.iter().filter_map(|g| PcieLink::detect(g.index)).collect();
        let storage = StorageInfo::detect_all();
        let (has_igp, igp_name) = detect_igp();
        let cuda_version = detect_cuda_version();

        let overhead_per_gpu = 512 * 1024 * 1024u64;
        let total_usable_vram: u64 = gpus.iter()
            .map(|g| g.free_vram.saturating_sub(overhead_per_gpu))
            .sum();
        let available_ram = memory.mem_available_kb * 1024;

        Self {
            cpu, memory, gpus, pcie, storage, has_igp, igp_name, cuda_version,
            total_usable_vram, available_ram,
        }
    }

    /// Print complete hardware profile to stderr.
    pub fn print_report(&self) {
        eprintln!("╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║                   HARDWARE PROFILE                          ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");

        // CPU
        eprintln!("║ CPU: {}", self.cpu.model_name);
        eprintln!("║   Cores: {} physical, {} logical, {} socket(s)",
            self.cpu.physical_cores, self.cpu.logical_cores, self.cpu.sockets);
        eprintln!("║   Clock: {:.0} MHz", self.cpu.base_mhz);
        eprintln!("║   SIMD:  {} (best tier)", self.cpu.best_simd());
        eprintln!("║   Cache: L1d={} KB  L2={} KB  L3={} KB",
            self.cpu.cache_l1d / 1024, self.cpu.cache_l2 / 1024, self.cpu.cache_l3 / 1024);

        // Memory
        let (total_ram_bytes, _) = detect_ram();
        eprintln!("║ RAM: {:.1} GB total, {:.1} GB available",
            total_ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            self.available_ram as f64 / (1024.0 * 1024.0 * 1024.0));
        eprintln!("║   Swap: {:.1} GB",
            self.memory.swap_total_kb as f64 / (1024.0 * 1024.0));

        // GPUs
        if self.gpus.is_empty() {
            eprintln!("║ GPU: None detected");
        } else {
            for (i, gpu) in self.gpus.iter().enumerate() {
                let pcie_info = self.pcie.iter().find(|p| p.gpu_index == gpu.index);
                let pcie_str = pcie_info
                    .map(|p| format!("PCIe Gen{} x{} ({:.0} MB/s)", p.gen, p.width, p.bandwidth_mbs))
                    .unwrap_or_else(|| "PCIe unknown".to_string());
                eprintln!("║ GPU{i}: {} — {:.1} GB VRAM ({:.1} GB free)",
                    gpu.name,
                    gpu.total_vram as f64 / (1024.0 * 1024.0 * 1024.0),
                    gpu.free_vram as f64 / (1024.0 * 1024.0 * 1024.0));
                eprintln!("║   Compute: sm_{}.{}  {pcie_str}", gpu.compute_major, gpu.compute_minor);
            }
        }
        if self.has_igp {
            eprintln!("║ IGP: {} (using system RAM)",
                self.igp_name.as_deref().unwrap_or("detected"));
        }

        // Storage
        for s in &self.storage {
            let kind = if s.rotational { "HDD" } else if s.transport == "nvme" { "NVMe" } else { "SSD" };
            eprintln!("║ Storage: {} ({kind}, {})", s.device, s.transport);
        }

        if let Some(ref cv) = self.cuda_version {
            eprintln!("║ CUDA: {cv}");
        }

        eprintln!("╚══════════════════════════════════════════════════════════════╝");
    }
}

// ---------------------------------------------------------------------------
// Execution plan — maps model to hardware
// ---------------------------------------------------------------------------

/// Where a layer runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceTarget {
    Cpu,
    Gpu(usize),
}

impl std::fmt::Display for DeviceTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceTarget::Cpu => write!(f, "CPU"),
            DeviceTarget::Gpu(i) => write!(f, "GPU{i}"),
        }
    }
}

/// Layer assignment in the execution plan.
#[derive(Debug, Clone)]
pub struct LayerAssignment {
    pub layer_idx: usize,
    pub device: DeviceTarget,
    /// Estimated weight memory for this layer (bytes).
    pub weight_bytes: u64,
    /// Layer kind description.
    pub kind: String,
}

/// Complete execution plan.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Per-layer assignments.
    pub layers: Vec<LayerAssignment>,
    /// Summary: layers per device.
    pub device_layer_count: HashMap<String, usize>,
    /// Estimated peak CPU RAM usage (bytes).
    pub cpu_ram_bytes: u64,
    /// Estimated peak GPU VRAM usage per device (bytes).
    pub gpu_vram_bytes: HashMap<usize, u64>,
    /// Whether TurboQuant KV cache is recommended.
    pub use_turboquant: bool,
    /// Recommended thread count for CPU layers.
    pub cpu_threads: usize,
    /// Estimated token/s (rough).
    pub estimated_tok_s: f64,
    /// Advisory messages.
    pub advisories: Vec<String>,
}

impl ExecutionPlan {
    /// Build an execution plan from hardware profile and model config.
    ///
    /// Strategy for constrained hardware (PCIe Gen1, 2× 1050 Ti):
    /// - GPU layers must be **persistent** (loaded once, kept in VRAM)
    /// - PCIe bandwidth is too low for per-token transfers
    /// - Pack as many layers as possible onto GPUs, rest on CPU
    /// - MoE layers are large (256 experts) but only 8 active → CPU-friendly (mmap pages in)
    /// - Attention layers are smaller → best candidates for GPU
    pub fn build(hw: &HardwareProfile, model_layers: usize, bytes_per_layer: u64,
                 is_moe: bool, n_embd: usize, _n_kv_heads: usize, _head_dim: usize) -> Self {
        let mut plan = Self {
            layers: Vec::with_capacity(model_layers),
            device_layer_count: HashMap::new(),
            cpu_ram_bytes: 0,
            gpu_vram_bytes: HashMap::new(),
            use_turboquant: false,
            cpu_threads: hw.cpu.physical_cores.max(1),
            estimated_tok_s: 0.0,
            advisories: Vec::new(),
        };

        // --- Compute per-GPU available VRAM ---
        let overhead_per_gpu = 512 * 1024 * 1024u64;
        let mut gpu_budgets: Vec<(usize, u64)> = hw.gpus.iter().map(|g| {
            (g.index, g.free_vram.saturating_sub(overhead_per_gpu))
        }).collect();
        // Sort by VRAM descending, then by PCIe bandwidth descending
        gpu_budgets.sort_by(|a, b| b.1.cmp(&a.1));

        // --- Estimate per-layer size ---
        // For MoE models: attention layers and FFN layers have very different sizes.
        // Attention: Q+K+V+O = 4 * n_embd * (n_heads * head_dim) * quant_ratio
        // MoE FFN: n_experts * 3 * n_embd * ffn_dim * quant_ratio (huge but mmap-friendly)
        // SSM: similar to attention in weight size
        //
        // Strategy: put attention/SSM layers on GPU (smaller, compute-bound)
        //           put MoE layers on CPU (huge, but only 8/256 experts touched per token = mmap-friendly)

        let total_vram: u64 = gpu_budgets.iter().map(|(_, v)| v).sum();
        let gpu_layers = if bytes_per_layer > 0 {
            (total_vram / bytes_per_layer).min(model_layers as u64) as usize
        } else {
            0
        };

        // Assign layers: first N go to GPUs (round-robin across GPUs)
        let mut gpu_used: HashMap<usize, u64> = HashMap::new();
        for layer_idx in 0..model_layers {
            // Try to assign to a GPU with remaining budget
            let mut assigned = DeviceTarget::Cpu;
            if layer_idx < gpu_layers {
                for &(gpu_idx, budget) in &gpu_budgets {
                    let used = gpu_used.get(&gpu_idx).copied().unwrap_or(0);
                    if used + bytes_per_layer <= budget {
                        assigned = DeviceTarget::Gpu(gpu_idx);
                        *gpu_used.entry(gpu_idx).or_insert(0) += bytes_per_layer;
                        break;
                    }
                }
            }

            let kind = if is_moe {
                if layer_idx % 2 == 0 { "SSM/Attention" } else { "MoE FFN" }
            } else {
                "Transformer"
            };

            plan.layers.push(LayerAssignment {
                layer_idx,
                device: assigned,
                weight_bytes: bytes_per_layer,
                kind: kind.to_string(),
            });

            let dev_name = assigned.to_string();
            *plan.device_layer_count.entry(dev_name).or_insert(0) += 1;
        }

        // GPU VRAM summary
        plan.gpu_vram_bytes = gpu_used;

        // CPU RAM: model is mmap'd so actual RSS comes from page faults.
        // Each CPU layer that's accessed will page in. For MoE, only 8/256 experts page in.
        // Estimate working set: N_cpu_layers * working_bytes_per_layer
        let n_cpu_layers = plan.device_layer_count.get("CPU").copied().unwrap_or(0) as u64;
        let cpu_working_bytes_per_layer = if is_moe {
            // MoE: only 8/256 experts loaded + attention = ~1/32 of full layer
            bytes_per_layer / 32 + bytes_per_layer / 4 // shared expert + attention parts
        } else {
            bytes_per_layer
        };
        plan.cpu_ram_bytes = n_cpu_layers * cpu_working_bytes_per_layer;

        // TurboQuant recommendation: always for MoE models, or when context > 1024
        plan.use_turboquant = is_moe;

        // --- Performance estimation ---
        let cpu_gops = hw.cpu.estimated_matvec_gops() * hw.cpu.physical_cores as f64;
        // Q8_0 matvec: n_embd * n_output ops per projection
        // Per attention layer: ~4 projections × n_embd² = 4 * n_embd² ops
        let ops_per_attn_layer = 4.0 * (n_embd as f64).powi(2) / 1e9; // in GOPS
        let ops_per_moe_layer = if is_moe {
            // 8 experts × 3 projections × n_embd * ffn_dim
            8.0 * 3.0 * n_embd as f64 * 512.0 / 1e9 // ffn_dim is typically small for MoE
        } else {
            3.0 * n_embd as f64 * (n_embd as f64 * 4.0) / 1e9
        };
        let total_ops = model_layers as f64 * (ops_per_attn_layer + ops_per_moe_layer);
        plan.estimated_tok_s = if total_ops > 0.0 { cpu_gops / total_ops } else { 0.0 };

        // --- Advisories ---
        if hw.gpus.is_empty() {
            plan.advisories.push("No GPUs detected. Running CPU-only.".to_string());
        }

        for link in &hw.pcie {
            if link.gen < 3 {
                plan.advisories.push(format!(
                    "GPU{}: PCIe Gen{} x{} ({:.0} MB/s) — SLOW. \
                     Weights must be persistent in VRAM (no per-token transfers).",
                    link.gpu_index, link.gen, link.width, link.bandwidth_mbs));
            }
        }

        if !hw.cpu.avx2 {
            plan.advisories.push(format!(
                "CPU lacks AVX2 — using {} ({}× slower than AVX2+FMA). \
                 GPU offloading is especially important.",
                hw.cpu.best_simd(),
                (8.0 / hw.cpu.estimated_matvec_gops()).round() as u32));
        }

        if hw.has_igp {
            plan.advisories.push(format!(
                "IGP ({}) is consuming system RAM for display framebuffer. \
                 Consider disabling in BIOS to free ~256-512 MB.",
                hw.igp_name.as_deref().unwrap_or("Intel HD")));
        }

        let headroom_gb = hw.memory.commit_headroom_kb() as f64 / (1024.0 * 1024.0);
        if headroom_gb < 2.0 && hw.memory.overcommit_mode == 0 {
            plan.advisories.push(format!(
                "Commit headroom only {headroom_gb:.1} GB. Risk of OOM. \
                 Run: sudo sysctl -w vm.overcommit_memory=1"));
        }

        if hw.cpu.physical_cores <= 2 {
            plan.advisories.push(
                "Only 2 CPU cores — limited parallelism. GPU offloading is critical.".to_string());
        }

        plan
    }

    /// Print execution plan to stderr.
    pub fn print_report(&self) {
        eprintln!("╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║                   EXECUTION PLAN                            ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");

        // Device summary
        for (dev, count) in &self.device_layer_count {
            eprintln!("║ {dev}: {count} layers");
        }
        if !self.gpu_vram_bytes.is_empty() {
            for (gpu_idx, bytes) in &self.gpu_vram_bytes {
                eprintln!("║   GPU{gpu_idx} VRAM: {:.2} GB allocated",
                    *bytes as f64 / (1024.0 * 1024.0 * 1024.0));
            }
        }
        eprintln!("║ CPU working set: {:.2} GB estimated",
            self.cpu_ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
        eprintln!("║ CPU threads: {}", self.cpu_threads);
        eprintln!("║ TurboQuant KV: {}", if self.use_turboquant { "ENABLED" } else { "disabled" });
        eprintln!("║ Est. throughput: {:.2} tok/s", self.estimated_tok_s);

        // Layer map (condensed)
        let mut ranges: Vec<(DeviceTarget, usize, usize)> = Vec::new();
        for la in &self.layers {
            if let Some(last) = ranges.last_mut() {
                if last.0 == la.device {
                    last.2 = la.layer_idx;
                    continue;
                }
            }
            ranges.push((la.device, la.layer_idx, la.layer_idx));
        }
        eprintln!("║ Layer map:");
        for (dev, start, end) in &ranges {
            if start == end {
                eprintln!("║   Layer {start}: {dev}");
            } else {
                eprintln!("║   Layers {start}-{end}: {dev}");
            }
        }

        // Advisories
        if !self.advisories.is_empty() {
            eprintln!("╠══════════════════════════════════════════════════════════════╣");
            for adv in &self.advisories {
                eprintln!("║ ⚠ {adv}");
            }
        }
        eprintln!("╚══════════════════════════════════════════════════════════════╝");
    }
}

// ---------------------------------------------------------------------------
// GPU detection (existing code below)
// ---------------------------------------------------------------------------

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
    /// Integrated GPU detected (Intel HD/UHD, AMD APU)
    pub has_igp: bool,
    /// IGP name if detected
    pub igp_name: Option<String>,
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
        let (has_igp, igp_name) = detect_igp();

        Self {
            gpus,
            total_ram,
            available_ram,
            cpu_cores,
            cuda_version,
            has_igp,
            igp_name,
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

        // IGP advisory
        if self.has_igp {
            if let Some(ref igp_name) = self.igp_name {
                println!("  IGP: {igp_name} detected");
                if !self.gpus.is_empty() {
                    println!(
                        "  TIP: Your IGP is using system RAM for display output. \
                         If you don't need display output from the IGP, disable it \
                         in BIOS to free up RAM and dedicate your discrete GPU(s) \
                         to compute."
                    );
                }
            }
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
                        .map(|v| v.split_whitespace().next().unwrap_or("").trim().to_string())
                })
        })
}

/// Detect integrated GPU (Intel HD/UHD/Iris, AMD APU) via lspci
fn detect_igp() -> (bool, Option<String>) {
    let output = Command::new("lspci").output();

    match output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            for line in stdout.lines() {
                let lower = line.to_lowercase();
                if lower.contains("vga") || lower.contains("3d") || lower.contains("display") {
                    // Intel IGP
                    if lower.contains("intel")
                        && (lower.contains("hd graphics")
                            || lower.contains("uhd graphics")
                            || lower.contains("iris"))
                    {
                        // Extract name after ": " and before " (rev"
                        let name = line
                            .split(": ")
                            .nth(1)
                            .unwrap_or("Intel IGP")
                            .split(" (rev")
                            .next()
                            .unwrap_or("Intel IGP")
                            .trim()
                            .to_string();
                        return (true, Some(name));
                    }
                    // AMD APU
                    if lower.contains("amd")
                        && lower.contains("radeon graphics")
                        && !lower.contains("rx ")
                    {
                        let name = line
                            .split(": ")
                            .nth(1)
                            .unwrap_or("AMD APU")
                            .split(" (rev")
                            .next()
                            .unwrap_or("AMD APU")
                            .trim()
                            .to_string();
                        return (true, Some(name));
                    }
                }
            }
            (false, None)
        }
        _ => (false, None),
    }
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
