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
