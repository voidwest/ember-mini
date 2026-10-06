//! GGUF reader. GGUF is the file format that holds the model.
//!
//! A GGUF file has four parts:
//! 1. A header: the magic bytes "GGUF", the version, and two counts.
//! 2. Metadata: key/value pairs, for example `gpt2.block_count = 12`.
//! 3. Tensor infos: the name, shape, data type and offset of each tensor.
//! 4. Tensor data: the raw bytes, which start at an aligned offset.
//!
//! All numbers are little-endian.
//!
//! EXERCISE 4. Write `Reader::string`, `Reader::value` and `load`. The other
//! parts are done. Check your work with `cargo test loader`.

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

    // TODO 1. The header:
    //   - 4 bytes: the magic "GGUF". If they are different, return an error.
    //   - u32: the version. Accept 2 and 3.
    //   - u64: the number of tensors.
    //   - u64: the number of metadata key/value pairs.
    // Hint: `return Err("message".into());` returns an error. `format!` works
    // too: `Err(format!("bad version {version}").into())`.

    // TODO 2. The metadata. For each pair, read:
    //   - a string: the key
    //   - u32: the value type
    //   - the value. Write `Reader::value` (below) for this.
    // Put the pairs into a `HashMap<String, Value>`.

    // TODO 3. The tensor infos. For each tensor, read:
    //   - a string: the name
    //   - u32: the number of dimensions
    //   - u64 for each dimension. GGUF writes the INNERMOST dimension first:
    //     a matrix of 2 rows of 3 numbers is written as [3, 2]. Reverse the
    //     list to get the row-major shape [2, 3] that `Tensor` uses.
    //   - u32: the data type (TYPE_F32, TYPE_F16 or TYPE_Q8_0)
    //   - u64: the offset of the data, from the start of the data section
    // Keep the infos in a list. The data comes after the last info.

    // TODO 4. The tensor data.
    //   - The data section starts at the first multiple of the alignment that
    //     is at or after `r.pos`. The alignment is the metadata value
    //     "general.alignment", or 32 if the file does not have it.
    //     Hint: `usize::div_ceil`.
    //   - The number of values in a tensor is the product of its shape.
    //   - The number of bytes: 4 for each F32 value, 2 for each F16 value, and
    //     `Q8_0_BYTES` for each block of `Q8_0_BLOCK` values in Q8_0. For all
    //     other types, return an error.
    //   - `bytes.get(start..start + n_bytes)` gives the bytes of one tensor,
    //     or `None` if the file is too short.
    //   - Convert the bytes with your functions in quant.rs, and put a
    //     `Tensor` into a `HashMap<String, Tensor>`.
    //
    // Then return `Ok(Gguf { metadata, tensors })`.
    todo!("load")
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
        // TODO: Read the u64 length, then take that many bytes.
        // Hint: `String::from_utf8_lossy(bytes).into_owned()` makes a String.
        todo!("string")
    }

    /// Reads one metadata value of the given GGUF value type.
    fn value(&mut self, value_type: u32) -> Result<Value> {
        // TODO: Read one value of type `value_type`:
        //
        //   type | data              | Value variant
        //   -----+-------------------+--------------
        //     0  | u8   (1 byte)     | UInt
        //     1  | i8   (1 byte)     | Int
        //     2  | u16  (2 bytes)    | UInt
        //     3  | i16  (2 bytes)    | Int
        //     4  | u32  (4 bytes)    | UInt
        //     5  | i32  (4 bytes)    | Int
        //     6  | f32  (4 bytes)    | Float
        //     7  | bool (1 byte)     | Bool (0 is false)
        //     8  | string            | Str
        //     9  | array             | Array
        //    10  | u64  (8 bytes)    | UInt
        //    11  | i64  (8 bytes)    | Int
        //    12  | f64  (8 bytes)    | Float
        //
        // An array is: a u32 item type, a u64 count, then the items. Each item
        // is a value too, so `value` can call itself.
        //
        // Hint: `i16::from_le_bytes(self.take(2)?.try_into()?)` reads an i16.
        // Hint: read all types, also the ones that this engine does not use.
        // If you skip a value, the next key starts at the wrong byte.
        todo!("value")
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
