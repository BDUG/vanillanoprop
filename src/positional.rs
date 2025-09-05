use crate::math::Matrix;

/// Generates sinusoidal positional encodings as in Vaswani et al.
pub fn positional_encoding(seq_len: usize, model_dim: usize) -> Matrix {
    let mut enc = Matrix::zeros(seq_len, model_dim);
    for pos in 0..seq_len {
        for i in 0..model_dim {
            let angle = (pos as f32) / (10000f32.powf((2 * (i / 2)) as f32 / model_dim as f32));
            let val = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            enc.set(pos, i, val);
        }
    }
    enc
}

/// Apply rotary positional encoding to query and key matrices in-place.
/// Both `q` and `k` are expected to have shape `(seq_len, dim)` where `dim`
/// is even. `base_freq` controls the angular frequency of the rotation
/// (typically `10000.0`).
pub fn apply_rope(q: &mut Matrix, k: &mut Matrix, base_freq: f32) {
    assert_eq!(q.rows, k.rows);
    assert_eq!(q.cols, k.cols);
    let dim = q.cols;
    assert!(dim % 2 == 0, "RoPE requires an even dimension");
    let half = dim / 2;
    for pos in 0..q.rows {
        for i in 0..half {
            let theta = (pos as f32) / base_freq.powf(2.0 * i as f32 / dim as f32);
            let (sin, cos) = theta.sin_cos();
            let q_even = q.data[pos * dim + 2 * i];
            let q_odd = q.data[pos * dim + 2 * i + 1];
            q.data[pos * dim + 2 * i] = q_even * cos - q_odd * sin;
            q.data[pos * dim + 2 * i + 1] = q_even * sin + q_odd * cos;
            let k_even = k.data[pos * dim + 2 * i];
            let k_odd = k.data[pos * dim + 2 * i + 1];
            k.data[pos * dim + 2 * i] = k_even * cos - k_odd * sin;
            k.data[pos * dim + 2 * i + 1] = k_even * sin + k_odd * cos;
        }
    }
}
