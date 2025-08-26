use crate::math::Matrix;
use rand::Rng;
use crate::rng::rng_from_env;

/// Dropout layer that randomly zeros elements during training.
///
/// During the forward pass, each element of the input is kept with
/// probability `1 - p`. When an element is kept its value is scaled by
/// `1/(1 - p)` to preserve the expected activation ("inverted" dropout).
/// The generated mask is stored so that it can be reused during the
/// backward pass.
pub struct Dropout {
    mask: Vec<f32>,
    rng: rand::rngs::StdRng,
}

impl Dropout {
    /// Create a new dropout layer.
    pub fn new() -> Self {
        Self { mask: Vec::new(), rng: rng_from_env() }
    }

    /// Forward pass for dropout.
    ///
    /// * `x` - Input matrix.
    /// * `p` - Dropout probability (fraction of units to drop).
    /// * `train` - Whether the network is in training mode.
    ///
    /// Returns the output matrix after applying dropout. When `train` is
    /// `false` the input is returned unchanged.
    pub fn forward(&mut self, x: &Matrix, p: f32, train: bool) -> Matrix {
        if train {
            let mut out = Matrix::zeros(x.rows, x.cols);
            self.mask = vec![0.0; x.data.len()];
            let scale = if p < 1.0 { 1.0 / (1.0 - p) } else { 0.0 };
            for i in 0..x.data.len() {
                if self.rng.gen::<f32>() < p {
                    self.mask[i] = 0.0;
                    out.data[i] = 0.0;
                } else {
                    self.mask[i] = scale;
                    out.data[i] = x.data[i] * scale;
                }
            }
            out
        } else {
            self.mask = vec![1.0; x.data.len()];
            x.clone()
        }
    }

    /// Backward pass using the mask generated in `forward`.
    pub fn backward(&self, grad: &Matrix) -> Matrix {
        let mut grad_input = Matrix::zeros(grad.rows, grad.cols);
        for i in 0..grad.data.len() {
            grad_input.data[i] = grad.data[i] * self.mask[i];
        }
        grad_input
    }
}
