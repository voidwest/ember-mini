//! Conversion of stored weights to f32.
//!
//! This engine converts all weights to f32 when it loads the model. Real
//! engines keep the weights quantized and convert them inside the matmul,
//! which uses 4x less memory. This engine does not, so that the math stays
//! simple.
//!
//! EXERCISE 2. Write the three functions. Check your work with
//! `cargo test quant`.

use half::f16;

/// Q8_0 keeps numbers in blocks of 32. Each block has 34 bytes: a 2-byte f16
/// scale `d`, then 32 signed 8-bit values `q`. Each real value is `q * d`.
pub const Q8_0_BLOCK: usize = 32;
pub const Q8_0_BYTES: usize = 2 + Q8_0_BLOCK;

pub fn dequantize_q8_0(src: &[u8]) -> Vec<f32> {
    // TODO: For each block of `Q8_0_BYTES` bytes:
    //   1. Read the scale `d` from the first 2 bytes. It is an f16.
    //   2. Read the 32 bytes after it. Each byte is a signed 8-bit value `q`.
    //   3. Push `q * d` (as f32) for each of the 32 values.
    //
    // Hint: `src.as_chunks::<Q8_0_BYTES>()` gives the full blocks as arrays.
    // Hint: `byte as i8` reads a byte as a signed number (255 becomes -1).
    todo!("dequantize_q8_0")
}

pub fn f16_to_f32(src: &[u8]) -> Vec<f32> {
    // TODO: Each f16 is 2 bytes, little-endian.
    // Hint: `f16::from_le_bytes([b0, b1]).to_f32()`.
    todo!("f16_to_f32")
}

pub fn f32_from_bytes(src: &[u8]) -> Vec<f32> {
    // TODO: Each f32 is 4 bytes, little-endian.
    // Hint: `src.as_chunks::<4>()` gives `[u8; 4]` arrays, and
    // `f32::from_le_bytes` changes one array into an f32.
    todo!("f32_from_bytes")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_block_is_scale_times_values() {
        let mut block = vec![0u8; Q8_0_BYTES];
        block[..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        block[2] = (-2i8) as u8;
        block[3] = 4;
        let out = dequantize_q8_0(&block);
        assert_eq!(out.len(), 32);
        assert_eq!(&out[..3], &[-1.0, 2.0, 0.0]);
    }

    #[test]
    fn f16_bytes_become_f32() {
        // 1.0 is 0x3C00 and -2.0 is 0xC000 in f16, little-endian.
        assert_eq!(f16_to_f32(&[0x00, 0x3C, 0x00, 0xC0]), vec![1.0, -2.0]);
    }

    #[test]
    fn f32_bytes_become_f32() {
        let bytes = [1.5f32.to_le_bytes(), (-0.25f32).to_le_bytes()].concat();
        assert_eq!(f32_from_bytes(&bytes), vec![1.5, -0.25]);
    }
}
