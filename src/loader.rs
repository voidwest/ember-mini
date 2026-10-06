//! GGUF reader. GGUF is the file format that holds the model.
//!
//! A GGUF file has four parts:
//! 1. A header: the magic bytes "GGUF", the version, and two counts.
//! 2. Metadata: key/value pairs, for example `gpt2.block_count = 12`.
//! 3. Tensor infos: the name, shape, data type and offset of each tensor.
//! 4. Tensor data: the raw bytes, which start at an aligned offset.
//!
//! All numbers are little-endian.

use crate::quant;
use crate::Result;
use std::collections::HashMap;

// GGML data type ids. This engine reads only these three.
const TYPE_F32: u32 = 0;
const TYPE_F16: u32 = 1;
const TYPE_Q8_0: u32 = 8;

/// A metadata value. The engine reads only some of them (the arrays hold the
/// tokenizer vocabulary, which comes from tokenizer.json in this engine).
#[allow(dead_code)]
pub enum Value {
    UInt(u64),
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

pub struct Tensor {
    /// Row-major shape, outermost dimension first: a matrix of `rows` rows
    /// of `cols` numbers has the shape `[rows, cols]`.
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub struct Gguf {
    pub metadata: HashMap<String, Value>,
    pub tensors: HashMap<String, Tensor>,
}

impl Gguf {
    pub fn get_usize(&self, key: &str) -> Result<usize> {
        match self.metadata.get(key) {
            Some(Value::UInt(n)) => Ok(*n as usize),
            Some(Value::Int(n)) if *n >= 0 => Ok(*n as usize),
            _ => Err(format!("metadata key {key} is missing or is not a count").into()),
        }
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key) {
            Some(Value::Float(x)) => Some(*x as f32),
            _ => None,
        }
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(Value::Str(s)) => Some(s),
            _ => None,
        }
    }
}

/// Reads the full file into memory and converts every tensor to f32.
pub fn load(path: &str) -> Result<Gguf> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut r = Reader {
        bytes: &bytes,
        pos: 0,
    };

    // 1. Header.
    if r.take(4)? != b"GGUF" {
        return Err(format!("{path} is not a GGUF file").into());
    }
    let version = r.u32()?;
    if version != 2 && version != 3 {
        return Err(format!("unsupported GGUF version {version}").into());
    }
    let n_tensors = r.u64()?;
    let n_metadata = r.u64()?;

    // 2. Metadata.
    let mut metadata = HashMap::new();
    for _ in 0..n_metadata {
        let key = r.string()?;
        let value_type = r.u32()?;
        metadata.insert(key, r.value(value_type)?);
    }

    // 3. Tensor infos. The data for all tensors comes after the last info.
    let mut infos = Vec::new();
    for _ in 0..n_tensors {
        let name = r.string()?;
        let n_dims = r.u32()?;
        let mut shape = Vec::new();
        for _ in 0..n_dims {
            shape.push(r.u64()? as usize);
        }
        // GGUF writes the innermost dimension first. Reverse it to get the
        // usual row-major order.
        shape.reverse();
        let data_type = r.u32()?;
        let offset = r.u64()? as usize;
        infos.push((name, shape, data_type, offset));
    }

    // 4. Tensor data. It starts at the next multiple of the alignment, and
    // each tensor offset is relative to that start.
    let alignment = match metadata.get("general.alignment") {
        Some(Value::UInt(n)) => *n as usize,
        _ => 32,
    };
    let data_start = r.pos.div_ceil(alignment) * alignment;

    let mut tensors = HashMap::new();
    for (name, shape, data_type, offset) in infos {
        let n: usize = shape.iter().product();
        let n_bytes = match data_type {
            TYPE_F32 => n * 4,
            TYPE_F16 => n * 2,
            TYPE_Q8_0 => n / quant::Q8_0_BLOCK * quant::Q8_0_BYTES,
            other => {
                return Err(format!(
                    "tensor {name} has type {other}; this engine reads only F32, F16 and Q8_0"
                )
                .into())
            }
        };
        let start = data_start + offset;
        let raw = bytes
            .get(start..start + n_bytes)
            .ok_or_else(|| format!("tensor {name} goes past the end of the file"))?;
        let data = match data_type {
            TYPE_F32 => quant::f32_from_bytes(raw),
            TYPE_F16 => quant::f16_to_f32(raw),
            _ => quant::dequantize_q8_0(raw),
        };
        if data.len() != n {
            return Err(format!("tensor {name}: {n} values do not fill whole Q8_0 blocks").into());
        }
        tensors.insert(name, Tensor { shape, data });
    }

    Ok(Gguf { metadata, tensors })
}

/// Reads little-endian values from a byte slice, from the start to the end.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let out = self
            .bytes
            .get(self.pos..self.pos + n)
            .ok_or("unexpected end of the GGUF file")?;
        self.pos += n;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    /// A GGUF string is a u64 length, then that many UTF-8 bytes.
    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(len)?).into_owned())
    }

    /// Reads one metadata value of the given GGUF value type.
    fn value(&mut self, value_type: u32) -> Result<Value> {
        Ok(match value_type {
            0 => Value::UInt(self.take(1)?[0] as u64),
            1 => Value::Int(self.take(1)?[0] as i8 as i64),
            2 => Value::UInt(u16::from_le_bytes(self.take(2)?.try_into()?) as u64),
            3 => Value::Int(i16::from_le_bytes(self.take(2)?.try_into()?) as i64),
            4 => Value::UInt(self.u32()? as u64),
            5 => Value::Int(i32::from_le_bytes(self.take(4)?.try_into()?) as i64),
            6 => Value::Float(f32::from_le_bytes(self.take(4)?.try_into()?) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                // An array is: the type of its items, a u64 count, the items.
                let item_type = self.u32()?;
                let count = self.u64()?;
                let mut items = Vec::new();
                for _ in 0..count {
                    items.push(self.value(item_type)?);
                }
                Value::Array(items)
            }
            10 => Value::UInt(self.u64()?),
            11 => Value::Int(i64::from_le_bytes(self.take(8)?.try_into()?)),
            12 => Value::Float(f64::from_le_bytes(self.take(8)?.try_into()?)),
            other => return Err(format!("unknown GGUF value type {other}").into()),
        })
    }
}
