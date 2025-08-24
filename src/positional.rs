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
