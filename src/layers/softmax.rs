use super::layer::Layer;
use super::linear::LinearT;
use crate::math::Matrix;
use crate::tensor::Tensor;

/// Softmax activation layer without parameters.
pub struct SoftmaxT {
    out: Matrix,
}

impl SoftmaxT {
    pub fn new() -> Self {
        Self {
            out: Matrix::zeros(0, 0),
        }
    }

    fn forward_internal(x: &Matrix) -> Matrix {
        x.softmax()
    }

    fn backward_internal(&self, grad_out: &Matrix) -> Matrix {
        let mut grad = Matrix::zeros(grad_out.rows, grad_out.cols);
        for r in 0..grad_out.rows {
            let row_start = r * grad_out.cols;
            let row_grad = &grad_out.data[row_start..row_start + grad_out.cols];
            let row_out = &self.out.data[row_start..row_start + grad_out.cols];
            let mut dot = 0.0;
            for c in 0..grad_out.cols {
                dot += row_grad[c] * row_out[c];
            }
            for c in 0..grad_out.cols {
                grad.data[row_start + c] = row_out[c] * (row_grad[c] - dot);
            }
        }
        grad
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let m = Matrix::from_vec(x.shape[0], x.shape[1], x.data.clone());
        let data = Self::forward_internal(&m);
        Tensor::from_matrix(data)
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        let out = Self::forward_internal(x);
        self.out = out.clone();
        out
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        self.backward_internal(grad_out)
    }
}

impl Layer for SoftmaxT {
    fn forward(&self, x: &Tensor) -> Tensor {
        SoftmaxT::forward(self, x)
    }

    fn forward_train(&mut self, x: &Matrix) -> Matrix {
        SoftmaxT::forward_train(self, x)
    }

    fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        SoftmaxT::backward(self, grad_out)
    }

    fn zero_grad(&mut self) {}

    fn fa_update(&mut self, grad_out: &Matrix, _lr: f32) -> Matrix {
        self.backward_internal(grad_out)
    }

    fn adam_step(&mut self, _lr: f32, _beta1: f32, _beta2: f32, _eps: f32, _weight_decay: f32) {}

    fn parameters(&mut self) -> Vec<&mut LinearT> {
        Vec::new()
    }
}
