//! The math. Every function works on plain f32 slices.

/// `out = W x + b`. `W` has one row of `x.len()` numbers for each output.
pub fn matvec(w: &[f32], b: Option<&[f32]>, x: &[f32], out: &mut [f32]) {
    let cols = x.len();
    for (row, o) in out.iter_mut().enumerate() {
        *o = dot(&w[row * cols..(row + 1) * cols], x);
        if let Some(b) = b {
            *o += b[row];
        }
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Moves `x` to mean 0 and variance 1, then scales and shifts each value
/// with the learned `weight` and `bias`.
pub fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();
    for i in 0..x.len() {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

/// GELU with the tanh approximation, as in the original GPT-2.
pub fn gelu(x: &mut [f32]) {
    let c = (2.0 / std::f32::consts::PI).sqrt();
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + (c * (*v + 0.044715 * *v * *v * *v)).tanh());
    }
}

/// Changes scores into probabilities that add up to 1. The subtraction of the
/// maximum does not change the result, but it prevents overflow in `exp`.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// `x += y`. The model uses this for the residual connections.
pub fn add(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// The index of the largest value.
pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-5, "got {got:?}, want {want:?}");
        }
    }

    #[test]
    fn dot_multiplies_pairs_and_adds_them() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn matvec_uses_one_row_for_each_output() {
        // W = [[1, 2], [3, 4], [5, 6]], x = [1, 1], b = [0.5, 0, -1]
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0; 3];
        matvec(&w, Some(&[0.5, 0.0, -1.0]), &[1.0, 1.0], &mut out);
        assert_eq!(out, [3.5, 7.0, 10.0]);
        matvec(&w, None, &[1.0, 1.0], &mut out);
        assert_eq!(out, [3.0, 7.0, 11.0]);
    }

    #[test]
    fn layer_norm_gives_mean_0_and_variance_1() {
        let mut out = [0.0; 4];
        layer_norm(&[1.0, 2.0, 3.0, 4.0], &[1.0; 4], &[0.0; 4], 0.0, &mut out);
        assert_close(&out, &[-1.3416408, -0.4472136, 0.4472136, 1.3416408]);
    }

    #[test]
    fn layer_norm_then_scales_and_shifts() {
        let mut out = [0.0; 4];
        layer_norm(&[1.0, 2.0, 3.0, 4.0], &[2.0; 4], &[1.0; 4], 0.0, &mut out);
        assert_close(&out, &[-1.6832816, 0.1055728, 1.8944272, 3.6832816]);
    }

    #[test]
    fn gelu_matches_known_values() {
        let mut x = [0.0, 1.0, -1.0, 3.0];
        gelu(&mut x);
        assert_close(&x, &[0.0, 0.841192, -0.158808, 2.9963626]);
    }

    #[test]
    fn softmax_gives_probabilities() {
        let mut x = [1.0, 2.0, 3.0];
        softmax(&mut x);
        assert_close(&x, &[0.09003057, 0.24472847, 0.66524096]);
    }

    #[test]
    fn softmax_does_not_overflow_with_large_scores() {
        let mut x = [1000.0, 1000.0];
        softmax(&mut x);
        assert_close(&x, &[0.5, 0.5]);
    }

    #[test]
    fn add_adds_in_place() {
        let mut x = [1.0, 2.0];
        add(&mut x, &[3.0, 4.0]);
        assert_eq!(x, [4.0, 6.0]);
    }

    #[test]
    fn argmax_finds_the_first_largest_value() {
        assert_eq!(argmax(&[0.1, 3.0, -1.0, 3.0]), 1);
        assert_eq!(argmax(&[-5.0, -2.0, -9.0]), 1);
    }
}
