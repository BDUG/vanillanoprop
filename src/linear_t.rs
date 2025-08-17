use crate::autograd::Tensor;
use crate::math::Matrix;

pub struct LinearT {
    pub w: Tensor,
}

impl LinearT {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let data = Matrix::from_vec(
            in_dim,
            out_dim,
            (0..in_dim * out_dim)
                .map(|_| (rand::random::<f32>() - 0.5) * 0.02)
                .collect(),
        );
        Self {
            w: Tensor::from_matrix(data, true),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::matmul(x, &self.w)
    }
}
