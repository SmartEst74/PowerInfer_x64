//! Backend abstraction for CPU/GPU execution

use std::any::Any;

/// Error type for backend operations
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("Out of memory")]
    OutOfMemory,
    #[error("Invalid kernel: {0}")]
    InvalidKernel(String),
    #[error("CUDA error: {0}")]
    CudaError(String),
    #[error("Vulkan error: {0}")]
    VulkanError(String),
    #[error("Unsupported operation")]
    Unsupported,
}

/// Backend abstraction - represents a compute device (CPU, CUDA, Vulkan)
pub trait Backend: Send + Sync {
    /// Get backend name (e.g., "CPU", "CUDA", "Vulkan")
    fn name(&self) -> &str;

    /// Allocate memory on the backend
    fn allocate(&self, size: usize) -> Result<Buffer, BackendError>;

    /// Free memory
    fn free(&self, buffer: Buffer) -> Result<(), BackendError>;

    /// Copy data from host to device (or within device)
    fn copy_to_device(&self, dst: &Buffer, src: &[u8], offset: usize) -> Result<(), BackendError>;

    /// Copy data from device to host
    fn copy_to_host(&self, src: &Buffer, dst: &mut [u8], offset: usize)
        -> Result<(), BackendError>;

    /// Launch a kernel
    fn launch_kernel(
        &self,
        kernel: &Kernel,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        args: &[KernelArg],
    ) -> Result<(), BackendError>;

    /// Synchronize the backend (wait for all operations to complete)
    fn synchronize(&self) -> Result<(), BackendError>;

    /// Get device memory info (free, total) in bytes
    fn memory_info(&self) -> Result<(usize, usize), BackendError>;

    /// As Any for downcasting
    fn as_any(&self) -> &dyn Any;
}

/// GPU memory buffer handle
#[derive(Debug, Clone)]
pub struct Buffer {
    pub id: u64,
    pub size: usize,
    pub backend_type: BackendType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Cpu,
    Cuda,
    Vulkan,
}

/// Kernel handle (compiled GPU kernel)
#[derive(Debug)]
pub struct Kernel {
    pub name: String,
    pub backend_type: BackendType,
}

/// Kernel argument types
#[derive(Debug)]
pub enum KernelArg<'a> {
    Buffer(&'a Buffer),
    Value(u64),
}

// Helper trait for downcasting backends
pub trait BackendDowncast: Backend {
    fn as_cpu(&self) -> Option<&CpuBackend> {
        None
    }
    fn as_cuda(&self) -> Option<&CudaBackend> {
        None
    }
    fn as_vulkan(&self) -> Option<&VulkanBackend> {
        None
    }
}

// Implement for all backends that provide casting
impl BackendDowncast for dyn Backend {}

// Factory for creating backends based on configuration
pub struct BackendFactory;

impl BackendFactory {
    /// Create CPU backend
    pub fn cpu() -> Box<dyn Backend> {
        Box::new(CpuBackend::new())
    }

    /// Create CUDA backend (if available)
    pub fn cuda(device_id: usize) -> Result<Box<dyn Backend>, BackendError> {
        let b = CudaBackend::new(device_id)?;
        Ok(Box::new(b))
    }

    /// Create Vulkan backend
    pub fn vulkan() -> Result<Box<dyn Backend>, BackendError> {
        let b = VulkanBackend::new()?;
        Ok(Box::new(b))
    }
}

// Placeholder CPU backend (will be fully implemented later)
pub struct CpuBackend {
    name: String,
    // CPU-specific state
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            name: "CPU".to_string(),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn allocate(&self, size: usize) -> Result<Buffer, BackendError> {
        // For CPU, allocate vec and leak it (unsafe but simple)
        // In real impl, use proper arena allocator
        Ok(Buffer {
            id: 0, // dummy
            size,
            backend_type: BackendType::Cpu,
        })
    }

    fn free(&self, _buffer: Buffer) -> Result<(), BackendError> {
        // In real impl, deallocate
        Ok(())
    }

    fn copy_to_device(
        &self,
        _dst: &Buffer,
        _src: &[u8],
        _offset: usize,
    ) -> Result<(), BackendError> {
        // CPU: just memcpy into allocated buffer
        Ok(())
    }

    fn copy_to_host(
        &self,
        _src: &Buffer,
        _dst: &mut [u8],
        _offset: usize,
    ) -> Result<(), BackendError> {
        Ok(())
    }

    fn launch_kernel(
        &self,
        kernel: &Kernel,
        _grid_dim: (u32, u32, u32),
        _block_dim: (u32, u32, u32),
        _args: &[KernelArg],
    ) -> Result<(), BackendError> {
        Err(BackendError::InvalidKernel(kernel.name.clone()))
    }

    fn synchronize(&self) -> Result<(), BackendError> {
        Ok(())
    }

    fn memory_info(&self) -> Result<(usize, usize), BackendError> {
        // Query system RAM - placeholder
        Ok((16 * 1024 * 1024 * 1024, 32 * 1024 * 1024 * 1024)) // 16/32 GB
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// CUDA backend — real implementation when cuda feature is enabled,
// placeholder stub otherwise.

#[cfg(feature = "cuda")]
pub struct CudaBackend {
    name: String,
    device_id: usize,
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    pub fn new(device_id: usize) -> Result<Self, BackendError> {
        // Validate CUDA is available by creating a context
        crate::cuda::cuda_impl::gpu_memory(device_id as u32)
            .map_err(|e| BackendError::CudaError(e))?;
        Ok(Self {
            name: format!("CUDA:{device_id}"),
            device_id,
        })
    }
}

#[cfg(feature = "cuda")]
impl Backend for CudaBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn allocate(&self, size: usize) -> Result<Buffer, BackendError> {
        Ok(Buffer {
            id: 0,
            size,
            backend_type: BackendType::Cuda,
        })
    }

    fn free(&self, _buffer: Buffer) -> Result<(), BackendError> {
        Ok(())
    }

    fn copy_to_device(
        &self,
        _dst: &Buffer,
        _src: &[u8],
        _offset: usize,
    ) -> Result<(), BackendError> {
        Ok(())
    }

    fn copy_to_host(
        &self,
        _src: &Buffer,
        _dst: &mut [u8],
        _offset: usize,
    ) -> Result<(), BackendError> {
        Ok(())
    }

    fn launch_kernel(
        &self,
        kernel: &Kernel,
        _grid: (u32, u32, u32),
        _block: (u32, u32, u32),
        _args: &[KernelArg],
    ) -> Result<(), BackendError> {
        Err(BackendError::InvalidKernel(kernel.name.clone()))
    }

    fn synchronize(&self) -> Result<(), BackendError> {
        Ok(())
    }

    fn memory_info(&self) -> Result<(usize, usize), BackendError> {
        crate::cuda::cuda_impl::gpu_memory(self.device_id as u32)
            .map_err(|e| BackendError::CudaError(e))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Placeholder CUDA backend when cuda feature is not enabled
#[cfg(not(feature = "cuda"))]
pub struct CudaBackend {
    name: String,
    #[allow(dead_code)]
    device_id: usize,
}

#[cfg(not(feature = "cuda"))]
impl CudaBackend {
    pub fn new(device_id: usize) -> Result<Self, BackendError> {
        Ok(Self {
            name: format!("CUDA:{device_id}"),
            device_id,
        })
    }
}

#[cfg(not(feature = "cuda"))]
impl Backend for CudaBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn allocate(&self, size: usize) -> Result<Buffer, BackendError> {
        Ok(Buffer {
            id: 0,
            size,
            backend_type: BackendType::Cuda,
        })
    }

    fn free(&self, _buffer: Buffer) -> Result<(), BackendError> {
        Ok(())
    }

    fn copy_to_device(
        &self,
        _dst: &Buffer,
        _src: &[u8],
        _offset: usize,
    ) -> Result<(), BackendError> {
        Ok(())
    }

    fn copy_to_host(
        &self,
        _src: &Buffer,
        _dst: &mut [u8],
        _offset: usize,
    ) -> Result<(), BackendError> {
        Ok(())
    }

    fn launch_kernel(
        &self,
        kernel: &Kernel,
        _grid: (u32, u32, u32),
        _block: (u32, u32, u32),
        _args: &[KernelArg],
    ) -> Result<(), BackendError> {
        Err(BackendError::InvalidKernel(kernel.name.clone()))
    }

    fn synchronize(&self) -> Result<(), BackendError> {
        Ok(())
    }

    fn memory_info(&self) -> Result<(usize, usize), BackendError> {
        Ok((2 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Placeholder Vulkan backend
pub struct VulkanBackend;

impl VulkanBackend {
    pub fn new() -> Result<Self, BackendError> {
        Ok(Self)
    }
}

impl Backend for VulkanBackend {
    fn name(&self) -> &str {
        "Vulkan"
    }

    fn allocate(&self, size: usize) -> Result<Buffer, BackendError> {
        Ok(Buffer {
            id: 0,
            size,
            backend_type: BackendType::Vulkan,
        })
    }

    fn free(&self, _buffer: Buffer) -> Result<(), BackendError> {
        Ok(())
    }

    fn copy_to_device(
        &self,
        _dst: &Buffer,
        _src: &[u8],
        _offset: usize,
    ) -> Result<(), BackendError> {
        Ok(())
    }

    fn copy_to_host(
        &self,
        _src: &Buffer,
        _dst: &mut [u8],
        _offset: usize,
    ) -> Result<(), BackendError> {
        Ok(())
    }

    fn launch_kernel(
        &self,
        kernel: &Kernel,
        _grid_dim: (u32, u32, u32),
        _block_dim: (u32, u32, u32),
        _args: &[KernelArg],
    ) -> Result<(), BackendError> {
        Err(BackendError::InvalidKernel(kernel.name.clone()))
    }

    fn synchronize(&self) -> Result<(), BackendError> {
        Ok(())
    }

    fn memory_info(&self) -> Result<(usize, usize), BackendError> {
        Ok((4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_basic() {
        let backend = CpuBackend::new();
        assert_eq!(backend.name(), "CPU");
        let buf = backend.allocate(1024).unwrap();
        assert_eq!(buf.size, 1024);
    }
}
