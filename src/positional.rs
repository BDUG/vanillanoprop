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

/// Apply rotary positional embeddings to query and key matrices.
///
/// Both `q` and `k` are expected to have shape `(seq_len, dim)` where `dim`
/// is even. The rotation is performed in-place using the base frequency
/// `base_freq` (typically `10000.0`).
pub fn apply_rope(q: &mut Matrix, k: &mut Matrix, base_freq: f32) {
    assert_eq!(q.rows, k.rows);
    assert_eq!(q.cols, k.cols);
    let rows = q.rows;
    let dim = q.cols;
    assert_eq!(dim % 2, 0, "dimension must be even for RoPE");
    let half = dim / 2;
    for r in 0..rows {
        for i in 0..half {
            let freq = base_freq.powf(-2.0 * i as f32 / dim as f32);
            let theta = r as f32 * freq;
            let cos = theta.cos();
            let sin = theta.sin();
            let q1 = q.data[r * dim + 2 * i];
            let q2 = q.data[r * dim + 2 * i + 1];
            q.data[r * dim + 2 * i] = q1 * cos - q2 * sin;
            q.data[r * dim + 2 * i + 1] = q1 * sin + q2 * cos;
            let k1 = k.data[r * dim + 2 * i];
            let k2 = k.data[r * dim + 2 * i + 1];
            k.data[r * dim + 2 * i] = k1 * cos - k2 * sin;
            k.data[r * dim + 2 * i + 1] = k1 * sin + k2 * cos;
        }
    }
}
