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

#[cfg(test)]
mod tests {
    use super::*;

    fn write_string(f: &mut Vec<u8>, s: &str) {
        f.extend((s.len() as u64).to_le_bytes());
        f.extend(s.as_bytes());
    }

    /// Makes a small GGUF file with four metadata keys and two tensors.
    fn tiny_gguf() -> Vec<u8> {
        let mut f = Vec::new();
        f.extend(b"GGUF");
        f.extend(3u32.to_le_bytes()); // version
        f.extend(2u64.to_le_bytes()); // tensor count
        f.extend(4u64.to_le_bytes()); // metadata count

        write_string(&mut f, "general.architecture");
        f.extend(8u32.to_le_bytes()); // type: string
        write_string(&mut f, "gpt2");
        write_string(&mut f, "gpt2.block_count");
        f.extend(4u32.to_le_bytes()); // type: u32
        f.extend(12u32.to_le_bytes());
        write_string(&mut f, "gpt2.attention.layer_norm_epsilon");
        f.extend(6u32.to_le_bytes()); // type: f32
        f.extend(1e-5f32.to_le_bytes());
        write_string(&mut f, "tokenizer.ggml.tokens");
        f.extend(9u32.to_le_bytes()); // type: array
        f.extend(8u32.to_le_bytes()); // of strings
        f.extend(2u64.to_le_bytes()); // with 2 items
        write_string(&mut f, "hi");
        write_string(&mut f, "there");

        // Tensor "a": 2 rows of 3 F32 values. GGUF writes the innermost
        // dimension first, so the dimensions are 3, 2.
        write_string(&mut f, "a");
        f.extend(2u32.to_le_bytes());
        f.extend(3u64.to_le_bytes());
        f.extend(2u64.to_le_bytes());
        f.extend(TYPE_F32.to_le_bytes());
        f.extend(0u64.to_le_bytes()); // offset
                                      // Tensor "q": one Q8_0 block of 32 values. "a" uses 24 bytes, so
                                      // "q" starts at the next multiple of 32.
        write_string(&mut f, "q");
        f.extend(1u32.to_le_bytes());
        f.extend(32u64.to_le_bytes());
        f.extend(TYPE_Q8_0.to_le_bytes());
        f.extend(32u64.to_le_bytes()); // offset

        // The data starts at the next multiple of 32.
        while f.len() % 32 != 0 {
            f.push(0);
        }
        for i in 0..6 {
            f.extend((i as f32).to_le_bytes());
        }
        f.extend([0u8; 8]); // padding up to offset 32
        f.extend(half::f16::from_f32(0.5).to_le_bytes());
        f.extend(0u8..32);
        f
    }

    #[test]
    fn reads_metadata_and_tensors() {
        let name = format!("ember-mini-test-{}.gguf", std::process::id());
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, tiny_gguf()).unwrap();
        let gguf = load(path.to_str().unwrap());
        std::fs::remove_file(&path).unwrap();
        let gguf = gguf.unwrap();

        assert_eq!(gguf.get_str("general.architecture"), Some("gpt2"));
        assert_eq!(gguf.get_usize("gpt2.block_count").unwrap(), 12);
        assert_eq!(
            gguf.get_f32("gpt2.attention.layer_norm_epsilon"),
            Some(1e-5)
        );

        let a = &gguf.tensors["a"];
        assert_eq!(a.shape, [2, 3]);
        assert_eq!(a.data, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let q = &gguf.tensors["q"];
        assert_eq!(q.shape, [32]);
        assert_eq!(q.data[..3], [0.0, 0.5, 1.0]);
        assert_eq!(q.data[31], 15.5);
    }

    #[test]
    fn rejects_a_file_that_is_not_gguf() {
        let name = format!("ember-mini-test-bad-{}.gguf", std::process::id());
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, b"NOPE and some more bytes").unwrap();
        let result = load(path.to_str().unwrap());
        std::fs::remove_file(&path).unwrap();
        assert!(result.is_err());
    }
}
