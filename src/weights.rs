//! Model weight loading from GGUF
//!
//! Loads tensor data from GGUF files, dequantizing to f32 for CPU inference.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use anyhow::{anyhow, Result};

use crate::gguf::GgufFile;
use crate::quant::{self, QuantizationType};

/// A single weight tensor stored in f32
#[derive(Clone)]
pub struct WeightTensor {
    pub name: String,
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl WeightTensor {
    /// Number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// All model weights loaded into memory
pub struct Weights {
    tensors: BTreeMap<String, WeightTensor>,
}

impl Weights {
    /// Load all weights from a GGUF file, dequantizing to f32
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let file = File::open(gguf.path())?;
        let mut reader = BufReader::new(file);

        let tensors_info = gguf.tensors();
        let mut tensors = BTreeMap::new();

        for tensor_info in tensors_info {
            let name = tensor_info.name.clone();
            let shape: Vec<usize> = tensor_info.shape.iter().map(|&d| d as usize).collect();
            let data_offset = tensor_info.offset;

            // Determine quantization type from kind
            let ggml_type: gguf_rs::GGMLType = tensor_info
                .kind
                .try_into()
                .map_err(|_| anyhow!("Unknown GGML type for tensor '{name}'"))?;
            let qtype = QuantizationType::from_ggml_type(ggml_type)?;

            let total_elements: usize = shape.iter().product();

            if total_elements == 0 {
                continue;
            }

            // Read tensor data from file
            let data = read_tensor_data(&mut reader, data_offset, qtype, total_elements)?;

            tensors.insert(name.clone(), WeightTensor { name, data, shape });
        }

        Ok(Self { tensors })
    }

    /// Get a weight tensor by name
    pub fn get(&self, name: &str) -> Option<&WeightTensor> {
        self.tensors.get(name)
    }

    /// Get a weight tensor by name, with error
    pub fn get_required(&self, name: &str) -> Result<&WeightTensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| anyhow!("Required tensor '{name}' not found"))
    }

    /// Get f32 data for a tensor, reshaped as needed
    pub fn get_data(&self, name: &str) -> Result<&[f32]> {
        self.get_required(name).map(|t| t.data.as_slice())
    }

    /// List all tensor names
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Number of tensors
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Read tensor data from file and dequantize to f32
fn read_tensor_data(
    reader: &mut BufReader<File>,
    offset: u64,
    qtype: QuantizationType,
    total_elements: usize,
) -> Result<Vec<f32>> {
    reader.seek(SeekFrom::Start(offset))?;

    match qtype {
        QuantizationType::F32 => {
            let mut buf = vec![0u8; total_elements * 4];
            reader.read_exact(&mut buf)?;
            quant::dequantize(&buf, qtype, 1, total_elements)
        }
        QuantizationType::F16 => {
            let mut buf = vec![0u8; total_elements * 2];
            reader.read_exact(&mut buf)?;
            quant::dequantize(&buf, qtype, 1, total_elements)
        }
        _ => {
            let block_size = qtype.block_size_bytes();
            let vals_per_block = qtype.values_per_block();
            let total_blocks = total_elements.div_ceil(vals_per_block);
            let bytes_needed = total_blocks * block_size;
            let mut buf = vec![0u8; bytes_needed];
            reader.read_exact(&mut buf)?;

            // For quantized types, dequantize in chunks matching the block layout
            let rows = 1;
            let cols = (total_elements / vals_per_block) * vals_per_block;
            if cols == total_elements {
                quant::dequantize(&buf, qtype, rows, cols)
            } else {
                // Handle remainder elements as zeros
                let mut result = quant::dequantize(&buf, qtype, rows, cols)?;
                result.resize(total_elements, 0.0);
                Ok(result)
            }
        }
    }
}
