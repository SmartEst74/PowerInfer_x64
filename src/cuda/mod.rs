//! CUDA GPU backend — cust 0.3.2 API

#[cfg(feature = "cuda")]
pub mod cuda_impl {
    use cust::prelude::*;
    use std::sync::OnceLock;

    static CUDA_INIT: std::sync::Once = std::sync::Once::new();
    static mut CUDA_INIT_RESULT: Option<Result<(), String>> = None;

    fn ensure_init() -> Result<(), String> {
        CUDA_INIT.call_once(|| {
            let result = cust::init(CudaFlags::empty()).map_err(|e| format!("CUDA init: {e}"));
            unsafe {
                CUDA_INIT_RESULT = Some(result);
            }
        });
        unsafe { CUDA_INIT_RESULT.as_ref().unwrap().clone() }
    }

    fn make_context(device_id: u32) -> Result<(Context, Device), String> {
        ensure_init()?;
        let device =
            Device::get_device(device_id).map_err(|e| format!("Device {device_id}: {e}"))?;
        let ctx = Context::new(device).map_err(|e| format!("Context: {e}"))?;
        Ok((ctx, device))
    }

    /// CUDA matvec kernel PTX (sm_61 = GTX 1050 Ti)
    const MATVEC_PTX: &str = r#"
.version 8.0
.target sm_61
.address_size 64
.visible .entry matvec_kernel(
    .param .u64 param_y, .param .u64 param_x, .param .u64 param_w,
    .param .u32 param_n_out, .param .u32 param_n_in)
{
    .reg .pred %p<3>; .reg .f32 %f<10>; .reg .b32 %r<10>; .reg .b64 %rd<10>;
    ld.param.u64 %rd1, [param_y]; ld.param.u64 %rd2, [param_x]; ld.param.u64 %rd3, [param_w];
    ld.param.u32 %r1, [param_n_out]; ld.param.u32 %r2, [param_n_in];
    mov.u32 %r3, %tid.x; mov.u32 %r4, %ctaid.x; mov.u32 %r5, %ntid.x;
    mad.lo.u32 %r6, %r4, %r5, %r3;
    setp.ge.u32 %p1, %r6, %r1; @%p1 bra DONE;
    mov.f32 %f1, 0f00000000; mov.u32 %r7, 0;
LOOP:
    setp.ge.u32 %p2, %r7, %r2; @%p2 bra STORE;
    cvt.u64.u32 %rd4, %r7; shl.b64 %rd4, %rd4, 2; add.s64 %rd5, %rd2, %rd4;
    ld.global.f32 %f2, [%rd5];
    mul.lo.u32 %r8, %r6, %r2; add.u32 %r9, %r8, %r7;
    cvt.u64.u32 %rd6, %r9; shl.b64 %rd6, %rd6, 2; add.s64 %rd7, %rd3, %rd6;
    ld.global.f32 %f3, [%rd7];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r7, %r7, 1; bra LOOP;
STORE:
    cvt.u64.u32 %rd8, %r6; shl.b64 %rd8, %rd8, 2; add.s64 %rd9, %rd1, %rd8;
    st.global.f32 [%rd9], %f1;
DONE: ret;
}"#;

    /// GPU matvec: y = x @ W (row-major W)
    pub fn gpu_matvec(
        y: &mut [f32],
        x: &[f32],
        w: &[f32],
        n_out: usize,
        n_in: usize,
    ) -> Result<(), String> {
        let (_ctx, _device) = make_context(0)?;
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| format!("Stream: {e}"))?;

        let x_bytes = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        let w_bytes = unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) };

        let d_x = DeviceBuffer::from_slice(x_bytes).map_err(|e| format!("Upload x: {e}"))?;
        let d_w = DeviceBuffer::from_slice(w_bytes).map_err(|e| format!("Upload w: {e}"))?;
        let mut d_y: DeviceBuffer<u8> =
            DeviceBuffer::zeroed(n_out * 4).map_err(|e| format!("Alloc y: {e}"))?;

        let module = Module::from_ptx(MATVEC_PTX, &[]).map_err(|e| format!("PTX: {e}"))?;
        let func = module
            .get_function("matvec_kernel")
            .map_err(|e| format!("Func: {e}"))?;

        let block: u32 = 256;
        let grid: u32 = ((n_out as u32) + block - 1) / block;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_y.as_device_ptr(), d_x.as_device_ptr(), d_w.as_device_ptr(),
                n_out as u32, n_in as u32
            ))
            .map_err(|e| format!("Launch: {e}"))?;
        }

        stream.synchronize().map_err(|e| format!("Sync: {e}"))?;

        let y_bytes =
            unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut u8, y.len() * 4) };
        d_y.copy_to(y_bytes).map_err(|e| format!("Download: {e}"))?;
        Ok(())
    }

    /// Query GPU memory
    pub fn gpu_memory(device_id: u32) -> Result<(usize, usize), String> {
        ensure_init()?;
        let device = Device::get_device(device_id).map_err(|e| format!("Device: {e}"))?;
        let total = device.total_memory().map_err(|e| format!("Total: {e}"))?;
        Ok((total / 2, total)) // Conservative estimate: half available
    }
}
