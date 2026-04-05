use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use crate::gguf::GgufFile;
use crate::quant::{self, QuantizationType};
use anyhow::{anyhow, Result};

/// A weight tensor backed by a memory-mapped GGUF file.
///
/// The raw bytes are NOT copied into heap; instead they live in a shared `Mmap`
/// and each tensor holds an (offset, byte_len) window into it.
/// The OS pages weight data in/out on demand — a 35B-param model at Q8_0
/// (~14 GB) can run on a machine with only 22 GB RAM.
#[derive(Clone)]
pub struct WeightTensor {
    pub name: String,
    /// Shared memory map of the entire GGUF file
    mmap: Arc<Mmap>,
    /// Byte offset into the mmap where this tensor's data starts
    offset: usize,
    /// Number of bytes belonging to this tensor
    byte_len: usize,
    /// Logical shape [ne0, ne1, ne2, ...] where ne0 is fastest dimension
    pub shape: Vec<usize>,
    /// Quantization format of the raw bytes
    pub qtype: QuantizationType,
}

impl WeightTensor {
    /// Access the raw bytes of this tensor (zero-copy slice into the mmap).
    #[inline]
    pub fn raw(&self) -> &[u8] {
        &self.mmap[self.offset..self.offset + self.byte_len]
    }

    /// Total number of logical elements
    pub fn total_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn len(&self) -> usize {
        self.total_elements()
    }

    pub fn is_empty(&self) -> bool {
        self.total_elements() == 0
    }

    /// Dequantize the entire tensor to f32.
    ///
    /// Allocates a new Vec<f32> each call. For large tensors use sparingly.
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        let n = self.total_elements();
        if n == 0 {
            return Ok(Vec::new());
        }
        quant::dequantize(self.raw(), self.qtype, 1, n)
    }

    /// Dequantize one expert's weight matrix from a packed 3D expert tensor.
    ///
    /// For tensors shaped [ne0, ne1, n_experts]:
    /// Expert k's bytes start at `k * bytes_per_expert`.
    pub fn expert_to_f32(&self, expert_idx: usize) -> Result<Vec<f32>> {
        let (raw_slice, n_per_expert) = self.expert_raw_slice(expert_idx)?;
        quant::dequantize(raw_slice, self.qtype, 1, n_per_expert)
    }

    /// Get the raw byte slice and element count for one expert's weight matrix.
    ///
    /// Zero-copy — returns a slice into the mmap. Use with `quant::matvec_col_major`
    /// to avoid materializing the full f32 matrix.
    pub fn expert_raw_slice(&self, expert_idx: usize) -> Result<(&[u8], usize)> {
        if self.shape.len() < 3 {
            return Err(anyhow!(
                "expert_raw_slice requires 3D tensor, got {}D for {}",
                self.shape.len(),
                self.name
            ));
        }
        let ne0 = self.shape[0];
        let ne1 = self.shape[1];
        let n_experts = self.shape[2];
        if expert_idx >= n_experts {
            return Err(anyhow!(
                "expert_idx {expert_idx} >= n_experts {n_experts} for {}",
                self.name
            ));
        }
        let n_per_expert = ne0 * ne1;
        let bs = self.qtype.block_size_bytes();
        let vp = self.qtype.values_per_block();
        let bytes_per_expert = n_per_expert.div_ceil(vp) * bs;
        let start = expert_idx * bytes_per_expert;
        let end = start + bytes_per_expert;
        let raw = self.raw();
        if end > raw.len() {
            return Err(anyhow!(
                "expert {} bytes {}..{} out of range {} for {}",
                expert_idx,
                start,
                end,
                raw.len(),
                self.name
            ));
        }
        Ok((&raw[start..end], n_per_expert))
    }

    /// Dequantize a single embedding row (token_embd.weight [ne0=n_embd, ne1=n_vocab]).
    ///
    /// Token i's embedding spans flat positions [i*ne0 .. (i+1)*ne0].
    pub fn embedding_row_to_f32(&self, row_idx: usize) -> Result<Vec<f32>> {
        let ne0 = self.shape[0];
        let bs = self.qtype.block_size_bytes();
        let vp = self.qtype.values_per_block();
        let bytes_per_row = ne0.div_ceil(vp) * bs;
        let start = row_idx * bytes_per_row;
        let end = start + bytes_per_row;
        let raw = self.raw();
        if end > raw.len() {
            return Err(anyhow!(
                "row {} bytes {}..{} out of range {} for {}",
                row_idx,
                start,
                end,
                raw.len(),
                self.name
            ));
        }
        quant::dequantize(&raw[start..end], self.qtype, 1, ne0)
    }
}

pub struct Weights {
    tensors: BTreeMap<String, WeightTensor>,
}

impl Weights {
    /// Map the GGUF file and build a zero-copy index of all tensors.
    ///
    /// RAM usage is O(n_tensors * metadata) not O(total_weight_bytes).
    /// Weight bytes are paged in by the OS when accessed.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let path = gguf.path();
        let file = File::open(path)?;
        // SAFETY: we hold the file open for the lifetime of the Mmap.
        // We never write to the mmap slice.
        let mmap = unsafe { Mmap::map(&file)? };
        // Use default mmap advice (MADV_NORMAL).  The OS will use moderate
        // readahead and LRU eviction — the hot working set (~2 GB of
        // attention + active experts) stabilises in the page cache after a
        // few tokens, avoiding constant NVMe re-reads.
        //
        // Previous MADV_RANDOM was counter-productive: it disabled readahead
        // AND caused aggressive eviction, forcing re-reads every token.
        let mmap = Arc::new(mmap);

        let data_section_start = parse_header(path)?;

        let mut tensors = BTreeMap::new();
        for t in gguf.tensors() {
            let shape: Vec<usize> = t.shape.iter().map(|&d| d as usize).collect();
            let gt: gguf_rs::GGMLType = t.kind.try_into().map_err(|_| anyhow!("bad type"))?;
            let qt = QuantizationType::from_ggml_type(gt)?;
            let total: usize = shape.iter().product();
            if total == 0 {
                continue;
            }
            let byte_len = t.size as usize;
            let offset = (data_section_start + t.offset) as usize;
            if offset + byte_len > mmap.len() {
                return Err(anyhow!(
                    "tensor {} at offset {offset}+{byte_len} exceeds mmap size {}",
                    t.name,
                    mmap.len()
                ));
            }
            tensors.insert(
                t.name.clone(),
                WeightTensor {
                    name: t.name.clone(),
                    mmap: Arc::clone(&mmap),
                    offset,
                    byte_len,
                    shape,
                    qtype: qt,
                },
            );
        }
        Ok(Self { tensors })
    }

    /// Advise the OS to pre-read the given tensors into the page cache.
    ///
    /// This starts async readahead so that the first forward pass doesn't
    /// stall on NVMe I/O for every weight page.
    #[cfg(unix)]
    pub fn prefetch(&self, names: &[String]) {
        use memmap2::Advice;
        for name in names {
            if let Some(t) = self.tensors.get(name) {
                let slice = &t.mmap[t.offset..t.offset + t.byte_len];
                // SAFETY: advise_range is safe — it's a hint to the kernel.
                let _ = t.mmap.advise_range(Advice::WillNeed, t.offset, t.byte_len);
                let _ = slice; // ensure borrow
            }
        }
    }

    /// Lock named tensor pages in RAM so the kernel cannot evict them.
    /// Returns total bytes locked. Best-effort: failures are silently ignored.
    #[cfg(unix)]
    pub fn mlock(&self, names: &[String]) -> usize {
        let mut total = 0usize;
        for name in names {
            if let Some(t) = self.tensors.get(name) {
                let ptr = t.mmap[t.offset..].as_ptr();
                let len = t.byte_len;
                // SAFETY: mlock is safe — ptr/len point into a valid mmap region.
                let ret = unsafe { libc::mlock(ptr as *const libc::c_void, len) };
                if ret == 0 {
                    total += len;
                }
            }
        }
        total
    }

    #[cfg(not(unix))]
    pub fn mlock(&self, _names: &[String]) -> usize {
        0
    }

    pub fn get(&self, n: &str) -> Option<&WeightTensor> {
        self.tensors.get(n)
    }

    /// Dequantize a named tensor to f32.
    pub fn get_data(&self, n: &str) -> Result<Vec<f32>> {
        self.tensors
            .get(n)
            .ok_or_else(|| anyhow!("{n} not found"))?
            .to_f32()
    }

    /// Check if a tensor exists
    pub fn has(&self, n: &str) -> bool {
        self.tensors.contains_key(n)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Compute bytes for data section start from GGUF header.
fn parse_header(path: &Path) -> Result<u64> {
    let mut f = File::open(path)?;
    let mut b8 = [0u8; 8];
    let mut b4 = [0u8; 4];
    f.seek(SeekFrom::Start(8))?;
    f.read_exact(&mut b8)?;
    let nt = u64::from_le_bytes(b8);
    f.read_exact(&mut b8)?;
    let nm = u64::from_le_bytes(b8);
    for _ in 0..nm {
        f.read_exact(&mut b8)?;
        let kl = u64::from_le_bytes(b8);
        f.seek(SeekFrom::Current(i64::try_from(kl).unwrap_or(0)))?;
        f.read_exact(&mut b4)?;
        let vt = u32::from_le_bytes(b4);
        skip_val(&mut f, vt)?;
    }
    for _ in 0..nt {
        f.read_exact(&mut b8)?;
        f.seek(SeekFrom::Current(
            i64::try_from(u64::from_le_bytes(b8)).unwrap_or(0),
        ))?;
        f.read_exact(&mut b4)?;
        let nd = u32::from_le_bytes(b4);
        f.seek(SeekFrom::Current(i64::from(nd) * 8 + 12))?;
    }
    Ok((f.stream_position()? + 31) & !31)
}

fn skip_val(f: &mut File, vt: u32) -> Result<()> {
    let mut b8 = [0u8; 8];
    match vt {
        0..=1 | 7 => {
            f.seek(SeekFrom::Current(1))?;
        }
        2..=3 => {
            f.seek(SeekFrom::Current(2))?;
        }
        4..=6 => {
            f.seek(SeekFrom::Current(4))?;
        }
        8 => {
            f.read_exact(&mut b8)?;
            f.seek(SeekFrom::Current(
                i64::try_from(u64::from_le_bytes(b8)).unwrap_or(0),
            ))?;
        }
        9 => {
            let mut b4 = [0u8; 4];
            f.read_exact(&mut b4)?;
            let et = u32::from_le_bytes(b4);
            f.read_exact(&mut b8)?;
            let n = u64::from_le_bytes(b8);
            if et == 8 {
                for _ in 0..n {
                    f.read_exact(&mut b8)?;
                    f.seek(SeekFrom::Current(
                        i64::try_from(u64::from_le_bytes(b8)).unwrap_or(0),
                    ))?;
                }
            } else {
                let es: u64 = match et {
                    0..=1 | 7 => 1,
                    2..=3 => 2,
                    4..=6 => 4,
                    10..=12 => 8,
                    _ => 0,
                };
                f.seek(SeekFrom::Current(i64::try_from(n * es).unwrap_or(0)))?;
            }
        }
        10..=12 => {
            f.seek(SeekFrom::Current(8))?;
        }
        _ => {}
    }
    Ok(())
}

/// Compute the number of bytes a tensor occupies in the GGUF file.
pub fn tensor_byte_size(qt: QuantizationType, n_elements: usize) -> usize {
    match qt {
        QuantizationType::F32 => n_elements * 4,
        QuantizationType::F16 => n_elements * 2,
        _ => {
            let bs = qt.block_size_bytes();
            let vp = qt.values_per_block();
            n_elements.div_ceil(vp) * bs
        }
    }
}

/// Read raw bytes from GGUF at a given absolute offset (used by benchmarks).
pub fn read_raw_bytes(path: &Path, offset: u64, byte_count: usize) -> Result<Vec<u8>> {
    use std::io::BufReader;
    let f = File::open(path)?;
    let mut r = BufReader::new(f);
    r.seek(SeekFrom::Start(offset))?;
    let mut buf = vec![0u8; byte_count];
    r.read_exact(&mut buf)?;
    Ok(buf)
}
