//! The math. Every function works on plain f32 slices.
//!
//! EXERCISE 1. Write each function. Check your work with `cargo test ops`.

/// `out = W x + b`. `W` has one row of `x.len()` numbers for each output.
pub fn matvec(w: &[f32], b: Option<&[f32]>, x: &[f32], out: &mut [f32]) {
    // TODO: For each output `row`, calculate the dot product of row `row` of
    // `w` with `x`. If there is a bias, add `b[row]`.
    //
    // Hint: `w` is a flat slice in row-major order: row 0, then row 1, and so
    // on. Each row has `x.len()` numbers, so row `r` starts at `r * x.len()`.
    // Hint: use your `dot` function.
    todo!("matvec")
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    // TODO: Multiply the pairs `a[i] * b[i]`, then add all of them.
    // Hint: `a.iter().zip(b)` gives the pairs.
    todo!("dot")
}

/// Moves `x` to mean 0 and variance 1, then scales and shifts each value
/// with the learned `weight` and `bias`.
pub fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, out: &mut [f32]) {
    // TODO: With n = x.len():
    //   mean   = (sum of x[i]) / n
    //   var    = (sum of (x[i] - mean)^2) / n        (divide by n, not n - 1)
    //   out[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]
    //
    // Hint: `eps` is a small number (1e-5 in GPT-2). It prevents a division
    // by zero when all values are the same.
    todo!("layer_norm")
}

/// GELU with the tanh approximation, as in the original GPT-2.
pub fn gelu(x: &mut [f32]) {
    // TODO: Change each value v in place:
    //   gelu(v) = 0.5 * v * (1 + tanh(sqrt(2 / pi) * (v + 0.044715 * v^3)))
    //
    // Hint: `f32::tanh`, `f32::sqrt` and `std::f32::consts::PI`.
    todo!("gelu")
}

/// Changes scores into probabilities that add up to 1. The subtraction of the
/// maximum does not change the result, but it prevents overflow in `exp`.
pub fn softmax(x: &mut [f32]) {
    // TODO: Change each value in place:
    //   softmax(x)[i] = exp(x[i]) / (sum of exp(x[j]) for all j)
    //
    // Hint: `exp(1000.0)` is infinity in f32, and infinity / infinity is NaN.
    // Subtract the largest value from each x[i] before `exp`. The result does
    // not change, because the same factor goes into the top and the bottom
    // of the fraction. The test `softmax_does_not_overflow_with_large_scores`
    // checks this.
    todo!("softmax")
}

/// `x += y`. The model uses this for the residual connections.
pub fn add(x: &mut [f32], y: &[f32]) {
    // TODO: Add each `y[i]` to `x[i]`.
    todo!("add")
}

/// The index of the largest value.
pub fn argmax(x: &[f32]) -> usize {
    // TODO: Find the index of the largest value. If two values are equal,
    // return the first index.
    //
    // Hint: `f32` does not implement `Ord` (because of NaN), so
    // `x.iter().max()` does not compile. A loop that remembers the index of
    // the largest value so far is sufficient.
    todo!("argmax")
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
