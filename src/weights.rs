use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::gguf::GgufFile;
use crate::quant::{self, QuantizationType};
use anyhow::{anyhow, Result};

#[derive(Clone)]
pub struct WeightTensor {
    pub name: String,
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}
impl WeightTensor {
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

pub struct Weights {
    tensors: BTreeMap<String, WeightTensor>,
}

impl Weights {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let ds = parse_header(gguf.path())?;
        let mut r = BufReader::new(File::open(gguf.path())?);
        let mut tensors = BTreeMap::new();
        for t in gguf.tensors() {
            let shape: Vec<usize> = t.shape.iter().map(|&d| d as usize).collect();
            let gt: gguf_rs::GGMLType = t.kind.try_into().map_err(|_| anyhow!("bad type"))?;
            let qt = QuantizationType::from_ggml_type(gt)?;
            let total: usize = shape.iter().product();
            if total == 0 {
                continue;
            }
            let data = read_data(&mut r, ds + t.offset, qt, total)?;
            tensors.insert(
                t.name.clone(),
                WeightTensor {
                    name: t.name.clone(),
                    data,
                    shape,
                },
            );
        }
        Ok(Self { tensors })
    }
    pub fn get(&self, n: &str) -> Option<&WeightTensor> {
        self.tensors.get(n)
    }
    pub fn get_data(&self, n: &str) -> Result<&[f32]> {
        self.tensors
            .get(n)
            .map(|t| t.data.as_slice())
            .ok_or_else(|| anyhow!("{n} not found"))
    }
    pub fn len(&self) -> usize {
        self.tensors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

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
            // ARRAY: u32 elem_type + u64 count + elements
            let mut b4 = [0u8; 4];
            f.read_exact(&mut b4)?;
            let et = u32::from_le_bytes(b4);
            f.read_exact(&mut b8)?;
            let n = u64::from_le_bytes(b8);
            if et == 8 {
                // Array of strings: each has u64 length + bytes
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

fn read_data(
    r: &mut BufReader<File>,
    off: u64,
    qt: QuantizationType,
    n: usize,
) -> Result<Vec<f32>> {
    r.seek(SeekFrom::Start(off))?;
    match qt {
        QuantizationType::F32 => {
            let mut b = vec![0u8; n * 4];
            r.read_exact(&mut b)?;
            quant::dequantize(&b, qt, 1, n)
        }
        QuantizationType::F16 => {
            let mut b = vec![0u8; n * 2];
            r.read_exact(&mut b)?;
            quant::dequantize(&b, qt, 1, n)
        }
        _ => {
            let bs = qt.block_size_bytes();
            let vp = qt.values_per_block();
            let need = n.div_ceil(vp) * bs;
            let mut b = vec![0u8; need];
            let got = r.read(&mut b)?;
            b.truncate(got);
            let cols = (n / vp) * vp;
            if cols == n && got >= need {
                quant::dequantize(&b, qt, 1, cols)
            } else {
                let dc = (got / bs) * vp;
                let mut res = if dc > 0 {
                    quant::dequantize(&b, qt, 1, dc)?
                } else {
                    Vec::new()
                };
                res.resize(n, 0.0);
                Ok(res)
            }
        }
    }
}
