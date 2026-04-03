//! CUDA GPU backend — persistent device contexts with pre-uploaded weights.
//!
//! Uses the `cust` crate (CUDA driver API wrapper) for direct GPU control.
//! Each `GpuDevice` holds a CUDA context and a reusable stream.  Weight
//! buffers are uploaded once and kept resident in VRAM for the lifetime of
//! the device instance, avoiding per-token PCIe transfers.

#[cfg(feature = "cuda")]
pub mod cuda_impl {
    use cust::prelude::*;
    use std::sync::OnceLock;

    static CUDA_INIT: std::sync::Once = std::sync::Once::new();
    // Use OnceLock instead of static mut to avoid undefined behavior.
    static CUDA_INIT_RESULT: OnceLock<Result<(), String>> = OnceLock::new();

    fn ensure_init() -> Result<(), String> {
        CUDA_INIT.call_once(|| {
            let result = cust::init(CudaFlags::empty()).map_err(|e| format!("CUDA init: {e}"));
            let _ = CUDA_INIT_RESULT.set(result);
        });
        CUDA_INIT_RESULT.get().unwrap().clone()
    }

    /// Optimized CUDA matvec kernel PTX (sm_61 = GTX 1050 Ti)
    ///
    /// Computes y[row] = dot(W[row, :], x) for each row in [0, n_out).
    /// Uses warp-per-row: 8 warps/block, each warp handles one row.
    /// - Shared memory for x vector (broadcast across warps)
    /// - Coalesced weight reads (adjacent lanes read adjacent addresses)
    /// - Warp shuffle reduction (no shared memory needed for reduce)
    const MATVEC_PTX: &str = r#"
.version 6.0
.target sm_61
.address_size 64

.extern .shared .align 4 .b8 smem[];

.visible .entry matvec_kernel(
    .param .u64 param_y, .param .u64 param_x, .param .u64 param_w,
    .param .u32 param_n_out, .param .u32 param_n_in)
{
    .reg .pred %p<4>;
    .reg .f32 %f<6>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<12>;

    ld.param.u64 %rd1, [param_y];
    ld.param.u64 %rd2, [param_x];
    ld.param.u64 %rd3, [param_w];
    ld.param.u32 %r1, [param_n_out];
    ld.param.u32 %r2, [param_n_in];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;

    // Phase 1: Cooperatively load x into shared memory
    mov.u32 %r6, smem;
    mov.u32 %r7, %r3;
LOAD_X:
    setp.ge.u32 %p1, %r7, %r2;
    @%p1 bra LOAD_DONE;
    cvt.u64.u32 %rd4, %r7;
    shl.b64 %rd4, %rd4, 2;
    add.s64 %rd5, %rd2, %rd4;
    ld.global.f32 %f1, [%rd5];
    shl.b32 %r8, %r7, 2;
    add.u32 %r9, %r6, %r8;
    st.shared.f32 [%r9], %f1;
    add.u32 %r7, %r7, %r5;
    bra LOAD_X;
LOAD_DONE:
    bar.sync 0;

    // Phase 2: Warp-per-row dot product (8 warps/block, 32 lanes each)
    shr.u32 %r10, %r3, 5;
    and.b32 %r11, %r3, 31;
    shr.u32 %r12, %r5, 5;
    mad.lo.u32 %r13, %r4, %r12, %r10;

    setp.ge.u32 %p2, %r13, %r1;
    @%p2 bra EXIT;

    // W_row_ptr = w + row * n_in * sizeof(f32)
    mul.lo.u32 %r14, %r13, %r2;
    cvt.u64.u32 %rd6, %r14;
    shl.b64 %rd6, %rd6, 2;
    add.s64 %rd7, %rd3, %rd6;

    mov.f32 %f2, 0f00000000;
    mov.u32 %r7, %r11;
DOT:
    setp.ge.u32 %p3, %r7, %r2;
    @%p3 bra REDUCE;
    cvt.u64.u32 %rd8, %r7;
    shl.b64 %rd8, %rd8, 2;
    add.s64 %rd9, %rd7, %rd8;
    ld.global.f32 %f3, [%rd9];
    shl.b32 %r8, %r7, 2;
    add.u32 %r9, %r6, %r8;
    ld.shared.f32 %f4, [%r9];
    fma.rn.f32 %f2, %f3, %f4, %f2;
    add.u32 %r7, %r7, 32;
    bra DOT;

REDUCE:
    mov.b32 %r15, 0xFFFFFFFF;
    shfl.sync.down.b32 %f5, %f2, 16, 31, %r15;
    add.f32 %f2, %f2, %f5;
    shfl.sync.down.b32 %f5, %f2, 8, 31, %r15;
    add.f32 %f2, %f2, %f5;
    shfl.sync.down.b32 %f5, %f2, 4, 31, %r15;
    add.f32 %f2, %f2, %f5;
    shfl.sync.down.b32 %f5, %f2, 2, 31, %r15;
    add.f32 %f2, %f2, %f5;
    shfl.sync.down.b32 %f5, %f2, 1, 31, %r15;
    add.f32 %f2, %f2, %f5;

    setp.ne.u32 %p3, %r11, 0;
    @%p3 bra EXIT;
    cvt.u64.u32 %rd10, %r13;
    shl.b64 %rd10, %rd10, 2;
    add.s64 %rd11, %rd1, %rd10;
    st.global.f32 [%rd11], %f2;

EXIT:
    ret;
}"#;

    use std::cell::RefCell;

    /// (d_x capacity in f32 elements, d_x buffer, d_y capacity in f32 elements, d_y buffer)
    type ScratchBuffers = (usize, DeviceBuffer<u8>, usize, DeviceBuffer<u8>);

    /// Persistent GPU device holding a CUDA context, stream, and compiled kernel.
    ///
    /// Created once per physical GPU at model load time.  Weight buffers are
    /// uploaded via [`GpuDevice::upload_f32`] and kept resident until the
    /// device is dropped.
    pub struct GpuDevice {
        ctx: Context,
        stream: Stream,
        matvec_func: Function<'static>,
        // Keep module alive so the function pointer stays valid.
        _module: &'static Module,
        pub device_id: u32,
        /// Reusable scratch buffers to avoid per-call CUDA alloc/free.
        /// (d_x capacity in f32 elements, d_x buffer, d_y capacity in f32 elements, d_y buffer)
        scratch: RefCell<Option<ScratchBuffers>>,
    }

    // SAFETY: GpuDevice holds CUDA resources that are bound to the creating thread's
    // context. We ensure all CUDA calls happen on the owning thread. The Send+Sync
    // impls are required because InferenceContext (which owns GpuDevice) is Send.
    unsafe impl Send for GpuDevice {}
    unsafe impl Sync for GpuDevice {}

    /// Handle to a contiguous f32 buffer in GPU VRAM.
    pub struct GpuBuffer {
        inner: DeviceBuffer<u8>,
        /// Number of f32 elements.
        pub len: usize,
    }

    // SAFETY: GpuBuffer is just a typed wrapper around DeviceBuffer<u8>.
    unsafe impl Send for GpuBuffer {}
    unsafe impl Sync for GpuBuffer {}

    impl GpuDevice {
        /// Create a persistent GPU context on the given device.
        pub fn new(device_id: u32) -> Result<Self, String> {
            ensure_init()?;
            let device = Device::get_device(device_id)
                .map_err(|e| format!("Device {device_id}: {e}"))?;
            let ctx = Context::new(device)
                .map_err(|e| format!("Context for device {device_id}: {e}"))?;
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
                .map_err(|e| format!("Stream: {e}"))?;

            // Compile PTX once and leak the module so the function pointer is 'static.
            let module = Module::from_ptx(MATVEC_PTX, &[])
                .map_err(|e| format!("PTX compile: {e}"))?;
            let module: &'static Module = Box::leak(Box::new(module));
            let matvec_func = module
                .get_function("matvec_kernel")
                .map_err(|e| format!("get_function: {e}"))?;

            Ok(Self {
                ctx,
                stream,
                matvec_func,
                _module: module,
                device_id,
                scratch: RefCell::new(None),
            })
        }

        /// Set this device's CUDA context as the current context on this thread.
        fn make_current(&self) -> Result<(), String> {
            cust::context::CurrentContext::set_current(&self.ctx)
                .map_err(|e| format!("set_current GPU {}: {e}", self.device_id))
        }

        /// Upload an f32 slice to VRAM. Returns a handle that stays resident
        /// until dropped.
        pub fn upload_f32(&self, data: &[f32]) -> Result<GpuBuffer, String> {
            self.make_current()?;
            let bytes = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
            };
            let inner = DeviceBuffer::from_slice(bytes)
                .map_err(|e| format!("upload_f32: {e}"))?;
            Ok(GpuBuffer {
                inner,
                len: data.len(),
            })
        }

        /// GPU matvec: y = W @ x where W is [n_out × n_in] row-major f32 in
        /// VRAM and x is a host-side f32 slice.
        ///
        /// The weight buffer `w` must have been created by [`upload_f32`] and
        /// contain `n_out * n_in` elements.
        pub fn matvec(
            &self,
            y: &mut [f32],
            x: &[f32],
            w: &GpuBuffer,
            n_out: usize,
            n_in: usize,
        ) -> Result<(), String> {
            self.make_current()?;
            debug_assert_eq!(y.len(), n_out);
            debug_assert_eq!(x.len(), n_in);
            debug_assert_eq!(w.len, n_out * n_in);

            // Reuse or grow scratch buffers to avoid per-call CUDA alloc/free.
            let mut scratch = self.scratch.borrow_mut();
            let need_realloc = match scratch.as_ref() {
                Some((xc, _, yc, _)) => *xc < n_in || *yc < n_out,
                None => true,
            };
            if need_realloc {
                let new_x_cap = match scratch.as_ref() {
                    Some((xc, _, _, _)) => (*xc).max(n_in),
                    None => n_in,
                };
                let new_y_cap = match scratch.as_ref() {
                    Some((_, _, yc, _)) => (*yc).max(n_out),
                    None => n_out,
                };
                let dx = DeviceBuffer::<u8>::zeroed(new_x_cap * 4)
                    .map_err(|e| format!("alloc d_x: {e}"))?;
                let dy = DeviceBuffer::<u8>::zeroed(new_y_cap * 4)
                    .map_err(|e| format!("alloc d_y: {e}"))?;
                *scratch = Some((new_x_cap, dx, new_y_cap, dy));
            }
            let (_, d_x, _, d_y) = scratch.as_mut().unwrap();

            // Upload x: use raw CUDA memcpy (avoids alloc/free, just copies into existing buffer)
            let x_slice = unsafe {
                std::slice::from_raw_parts(x.as_ptr() as *const u8, n_in * 4)
            };
            // Upload x: use raw CUDA memcpy (avoids alloc/free, just copies into existing buffer)
            unsafe {
                let res = cust::sys::cuMemcpyHtoD_v2(
                    d_x.as_device_ptr().as_raw(),
                    x_slice.as_ptr() as *const std::ffi::c_void,
                    n_in * 4,
                );
                if res != cust::sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("upload x: {res:?}"));
                }
            }

            let block: u32 = 256;
            let rows_per_block: u32 = block / 32;
            let grid: u32 = (n_out as u32).div_ceil(rows_per_block);
            let smem_bytes: u32 = (n_in * 4) as u32;

            let stream = &self.stream;
            let func = &self.matvec_func;
            unsafe {
                launch!(func<<<grid, block, smem_bytes, stream>>>(
                    d_y.as_device_ptr(),
                    d_x.as_device_ptr(),
                    w.inner.as_device_ptr(),
                    n_out as u32,
                    n_in as u32
                ))
                .map_err(|e| format!("launch: {e}"))?;
            }

            self.stream.synchronize().map_err(|e| format!("sync: {e}"))?;

            // Download y: raw CUDA memcpy from device to host
            let y_bytes = unsafe {
                std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut u8, n_out * 4)
            };
            unsafe {
                let res = cust::sys::cuMemcpyDtoH_v2(
                    y_bytes.as_mut_ptr() as *mut std::ffi::c_void,
                    d_y.as_device_ptr().as_raw(),
                    n_out * 4,
                );
                if res != cust::sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("download y: {res:?}"));
                }
            }
            Ok(())
        }

        /// Query free and total VRAM in bytes.
        pub fn memory_info(&self) -> Result<(usize, usize), String> {
            self.make_current()?;
            // cust memory_get_info returns (free, total)
            let (free, total) = cust::memory::mem_get_info()
                .map_err(|e| format!("mem_get_info: {e}"))?;
            Ok((free, total))
        }
    }

    /// Query GPU memory (legacy, used by runtime/mod.rs).
    pub fn gpu_memory(device_id: u32) -> Result<(usize, usize), String> {
        ensure_init()?;
        let device = Device::get_device(device_id).map_err(|e| format!("Device: {e}"))?;
        let total = device.total_memory().map_err(|e| format!("Total: {e}"))?;
        Ok((total / 2, total)) // Conservative estimate: half available
    }

    use std::collections::HashMap;

    /// Pre-uploaded weight buffer on a specific GPU, keyed by weight name.
    struct LayerGpuWeights {
        device_idx: usize,
        /// weight_suffix → (GpuBuffer, n_out, n_in)
        /// suffix examples: "attn_qkv", "attn_gate", "ssm_out",
        ///                   "attn_q", "attn_k", "attn_v", "attn_output"
        buffers: HashMap<String, (GpuBuffer, usize, usize)>,
    }

    /// Part of the LM head weight matrix uploaded to one GPU.
    struct LmHeadPart {
        device_idx: usize,
        buffer: GpuBuffer,
        row_start: usize,
        n_rows: usize,
        n_in: usize,
    }

    /// Holds all GPU devices and pre-uploaded layer weights.
    ///
    /// Created once at model load time by [`GpuResources::init`].
    pub struct GpuResources {
        devices: Vec<GpuDevice>,
        /// layer_idx → pre-uploaded weights
        layers: HashMap<usize, LayerGpuWeights>,
        /// LM head split across GPUs (None if not uploaded)
        lm_head: Option<Vec<LmHeadPart>>,
    }

    impl GpuResources {
        /// Initialise GPU devices and upload dequantized weights for assigned layers.
        ///
        /// `assignments` is a list of `(layer_idx, gpu_device_id)` pairs from the
        /// execution plan. `get_weight` is a callback that dequantizes the named
        /// weight tensor to f32 and returns `(data, n_rows, n_cols)`.
        pub fn init<F>(
            assignments: &[(usize, usize)],
            mut get_weight: F,
        ) -> Result<Self, String>
        where
            F: FnMut(usize, &str) -> Option<(Vec<f32>, usize, usize)>,
        {
            // Determine unique GPU ids.
            let mut gpu_ids: Vec<usize> = assignments.iter().map(|a| a.1).collect();
            gpu_ids.sort_unstable();
            gpu_ids.dedup();

            let mut devices = Vec::new();
            let mut id_to_idx = HashMap::new();
            for &gid in &gpu_ids {
                let dev = GpuDevice::new(gid as u32)?;
                let (free, total) = dev.memory_info().unwrap_or((0, 0));
                eprintln!(
                    "[GPU] Device {gid}: {:.0} MB free / {:.0} MB total",
                    free as f64 / 1e6,
                    total as f64 / 1e6
                );
                id_to_idx.insert(gid, devices.len());
                devices.push(dev);
            }

            let mut layers = HashMap::new();

            // Weight suffixes to try uploading per layer.
            // SSM/GDR layers have attn_qkv + attn_gate + ssm_out.
            // Full attention layers have attn_q + attn_k + attn_v + attn_output.
            let weight_suffixes = [
                "attn_qkv", "attn_gate", "ssm_out",
                "attn_q", "attn_k", "attn_v", "attn_output",
            ];

            for &(layer_idx, gpu_id) in assignments {
                let dev_idx = *id_to_idx.get(&gpu_id).unwrap();
                let mut buffers = HashMap::new();

                for suffix in &weight_suffixes {
                    if let Some((data, n_out, n_in)) = get_weight(layer_idx, suffix) {
                        let buf = devices[dev_idx].upload_f32(&data)?;
                        let mb = (data.len() * 4) as f64 / 1e6;
                        eprintln!(
                            "[GPU] Uploaded blk.{layer_idx}.{suffix}.weight ({n_out}×{n_in}, {mb:.1} MB) → GPU {gpu_id}"
                        );
                        buffers.insert(suffix.to_string(), (buf, n_out, n_in));
                    }
                }

                if !buffers.is_empty() {
                    layers.insert(layer_idx, LayerGpuWeights {
                        device_idx: dev_idx,
                        buffers,
                    });
                }
            }

            let total_layers = layers.len();
            let total_buffers: usize = layers.values().map(|l| l.buffers.len()).sum();
            eprintln!("[GPU] Uploaded {total_buffers} weight tensors across {total_layers} layers");

            Ok(Self { devices, layers, lm_head: None })
        }

        /// Upload the LM head weight matrix, splitting across all GPUs.
        ///
        /// `data` is row-major f32 of shape [n_out × n_in].
        /// Each GPU gets a contiguous slice of output rows.
        pub fn upload_lm_head(&mut self, data: &[f32], n_out: usize, n_in: usize) -> Result<(), String> {
            let n_gpus = self.devices.len();
            if n_gpus == 0 {
                return Ok(());
            }
            let mut parts = Vec::new();
            let rows_per_gpu = n_out.div_ceil(n_gpus);
            for (dev_idx, dev) in self.devices.iter().enumerate() {
                let row_start = dev_idx * rows_per_gpu;
                let row_end = (row_start + rows_per_gpu).min(n_out);
                if row_start >= row_end {
                    break;
                }
                let n_rows = row_end - row_start;
                let slice = &data[row_start * n_in..row_end * n_in];
                let buffer = dev.upload_f32(slice)?;
                let mb = (slice.len() * 4) as f64 / 1e6;
                eprintln!(
                    "[GPU] Uploaded LM head rows {row_start}..{row_end} ({n_rows}×{n_in}, {mb:.1} MB) → GPU {}",
                    dev.device_id
                );
                parts.push(LmHeadPart { device_idx: dev_idx, buffer, row_start, n_rows, n_in });
            }
            self.lm_head = Some(parts);
            Ok(())
        }

        /// Run LM head matvec on GPU if uploaded. Returns None if not on GPU.
        ///
        /// Each GPU computes its subset of output rows, results are assembled
        /// into the full logits vector.
        pub fn try_lm_head_matvec(
            &self,
            logits: &mut [f32],
            x: &[f32],
        ) -> Option<Result<(), String>> {
            let parts = self.lm_head.as_ref()?;
            for part in parts {
                let dev = &self.devices[part.device_idx];
                let y_slice = &mut logits[part.row_start..part.row_start + part.n_rows];
                if let Err(e) = dev.matvec(y_slice, x, &part.buffer, part.n_rows, part.n_in) {
                    return Some(Err(e));
                }
            }
            Some(Ok(()))
        }

        /// Run matvec on GPU if the weight is uploaded; returns `None` if not a GPU layer.
        ///
        /// `y = W @ x` where W is `[n_out × n_in]` stored in VRAM.
        pub fn try_matvec(
            &self,
            y: &mut [f32],
            x: &[f32],
            layer_idx: usize,
            weight_suffix: &str,
        ) -> Option<Result<(), String>> {
            let lw = self.layers.get(&layer_idx)?;
            let (buf, n_out, n_in) = lw.buffers.get(weight_suffix)?;
            Some(self.devices[lw.device_idx].matvec(y, x, buf, *n_out, *n_in))
        }
    }
}
