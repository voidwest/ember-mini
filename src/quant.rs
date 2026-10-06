//! Conversion of stored weights to f32.
//!
//! This engine converts all weights to f32 when it loads the model. Real
//! engines keep the weights quantized and convert them inside the matmul,
//! which uses 4x less memory. This engine does not, so that the math stays
//! simple.

use half::f16;

/// Q8_0 keeps numbers in blocks of 32. Each block has 34 bytes: a 2-byte f16
/// scale `d`, then 32 signed 8-bit values `q`. Each real value is `q * d`.
pub const Q8_0_BLOCK: usize = 32;
pub const Q8_0_BYTES: usize = 2 + Q8_0_BLOCK;

pub fn dequantize_q8_0(src: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(src.len() / Q8_0_BYTES * Q8_0_BLOCK);
    let (blocks, _) = src.as_chunks::<Q8_0_BYTES>();
    for block in blocks {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        for &q in &block[2..] {
            out.push(q as i8 as f32 * d);
        }
    }
    out
}

pub fn f16_to_f32(src: &[u8]) -> Vec<f32> {
    let (values, _) = src.as_chunks::<2>();
    values
        .iter()
        .map(|&b| f16::from_le_bytes(b).to_f32())
        .collect()
}

pub fn f32_from_bytes(src: &[u8]) -> Vec<f32> {
    let (values, _) = src.as_chunks::<4>();
    values.iter().map(|&b| f32::from_le_bytes(b)).collect()
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
