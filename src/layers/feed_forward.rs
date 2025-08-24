use crate::autograd::Tensor;
use crate::math::Matrix;
use super::linear::LinearT;

pub struct FeedForwardT {
    pub w1: LinearT,
    pub w2: LinearT,
    // caches for backward
    mask: Vec<f32>,
    h1: Matrix,
}

impl FeedForwardT {
    pub fn new(dim: usize, hidden: usize) -> Self {
        Self {
            w1: LinearT::new(dim, hidden),
            w2: LinearT::new(hidden, dim),
            mask: Vec::new(),
            h1: Matrix::zeros(0, 0),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = self.w1.forward(x);
        for v in h.data.data.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        self.w2.forward(&h)
    }

    pub fn forward_local(&mut self, x: &Matrix) -> Matrix {
        let mut h1 = self.w1.forward_local(x);
        let mut mask = vec![0.0; h1.data.len()];
        for (i, v) in h1.data.iter_mut().enumerate() {
            if *v < 0.0 {
                *v = 0.0;
            } else {
                mask[i] = 1.0;
            }
        }
        self.mask = mask;
        let out = self.w2.forward_local(&h1);
        self.h1 = h1;
        out
    }

    pub fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        let grad_h1 = self.w2.fa_update(grad_out, lr);
        let mut g = grad_h1.clone();
        for (i, v) in g.data.iter_mut().enumerate() {
            *v *= self.mask[i];
        }
        self.w1.fa_update(&g, lr)
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        self.forward_local(x)
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let grad_h1 = self.w2.backward(grad_out);
        let mut g = grad_h1.clone();
        for (i, v) in g.data.iter_mut().enumerate() {
            *v *= self.mask[i];
        }
        self.w1.backward(&g)
    }

    pub fn zero_grad(&mut self) {
        self.w1.zero_grad();
        self.w2.zero_grad();
    }

    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.w1
            .adam_step(lr, beta1, beta2, eps, weight_decay);
        self.w2
            .adam_step(lr, beta1, beta2, eps, weight_decay);
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let (w1, w2) = (&mut self.w1, &mut self.w2);
        vec![w1, w2]
    }
}

