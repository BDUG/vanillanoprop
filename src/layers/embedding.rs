use crate::autograd::Tensor;
use crate::math::Matrix;
use super::linear::LinearT;

/// Embedding layer: maps one-hot (vocab_size) into dense model_dim.
pub struct EmbeddingT {
    pub table: LinearT, // weight matrix (vocab_size x model_dim)
}

impl EmbeddingT {
    pub fn new(vocab_size: usize, model_dim: usize) -> Self {
        Self {
            table: LinearT::new(vocab_size, model_dim),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // inference helper
        Tensor::matmul(x, &self.table.w)
    }

    pub fn forward_local(&mut self, x: &Matrix) -> Matrix {
        self.table.forward_local(x)
    }

    pub fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        self.table.fa_update(grad_out, lr)
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        self.table.forward_train(x)
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        self.table.backward(grad_out)
    }

    pub fn zero_grad(&mut self) {
        self.table.zero_grad();
    }

    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.table
            .adam_step(lr, beta1, beta2, eps, weight_decay);
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        self.table.parameters()
    }
}

