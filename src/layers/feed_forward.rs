use super::layer::Layer;
use super::linear::LinearT;
use super::{leaky_relu, relu, sigmoid, tanh};
use crate::math::Matrix;
use crate::tensor::Tensor;

#[derive(Clone, Copy)]
pub enum Activation {
    None,
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
}

pub struct FeedForwardT {
    pub w1: LinearT,
    pub w2: LinearT,
    pub activation: Activation,
    // caches for backward
    mask: Vec<f32>,
    h1: Matrix,
}

impl FeedForwardT {
    pub fn new(dim: usize, hidden: usize, activation: Activation) -> Self {
        Self {
            w1: LinearT::new(dim, hidden),
            w2: LinearT::new(hidden, dim),
            activation,
            mask: Vec::new(),
            h1: Matrix::zeros(0, 0),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = self.w1.forward(x);
        match self.activation {
            Activation::ReLU => relu::forward_tensor(&mut h),
            Activation::Sigmoid => sigmoid::forward_tensor(&mut h),
            Activation::Tanh => tanh::forward_tensor(&mut h),
            Activation::LeakyReLU => leaky_relu::forward_tensor(&mut h),
            Activation::None => {}
        }
        self.w2.forward(&h)
    }

    pub fn forward_local(&mut self, x: &Matrix) -> Matrix {
        let mut h1 = self.w1.forward_local(x);
        match self.activation {
            Activation::ReLU => {
                let mask = relu::forward_matrix(&mut h1);
                self.mask = mask;
            }
            Activation::Sigmoid => {
                sigmoid::forward_matrix(&mut h1);
                self.mask = vec![1.0; h1.data.len()];
            }
            Activation::Tanh => {
                tanh::forward_matrix(&mut h1);
                self.mask = vec![1.0; h1.data.len()];
            }
            Activation::LeakyReLU => {
                let mask = leaky_relu::forward_matrix(&mut h1);
                self.mask = mask;
            }
            Activation::None => {
                self.mask = vec![1.0; h1.data.len()];
            }
        }
        let out = self.w2.forward_local(&h1);
        self.h1 = h1;
        out
    }

    pub fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        let grad_h1 = self.w2.fa_update(grad_out, lr);
        let mut g = grad_h1.clone();
        match self.activation {
            Activation::ReLU => {
                for (i, v) in g.data.iter_mut().enumerate() {
                    *v *= self.mask[i];
                }
            }
            Activation::Sigmoid => {
                for (v, &h) in g.data.iter_mut().zip(self.h1.data.iter()) {
                    *v *= h * (1.0 - h);
                }
            }
            Activation::Tanh => {
                for (v, &h) in g.data.iter_mut().zip(self.h1.data.iter()) {
                    *v *= 1.0 - h * h;
                }
            }
            Activation::LeakyReLU => {
                for (i, v) in g.data.iter_mut().enumerate() {
                    *v *= self.mask[i];
                }
            }
            Activation::None => {}
        }
        self.w1.fa_update(&g, lr)
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        self.forward_local(x)
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let grad_h1 = self.w2.backward(grad_out);
        let mut g = grad_h1.clone();
        match self.activation {
            Activation::ReLU => {
                for (i, v) in g.data.iter_mut().enumerate() {
                    *v *= self.mask[i];
                }
            }
            Activation::Sigmoid => {
                for (v, &h) in g.data.iter_mut().zip(self.h1.data.iter()) {
                    *v *= h * (1.0 - h);
                }
            }
            Activation::Tanh => {
                for (v, &h) in g.data.iter_mut().zip(self.h1.data.iter()) {
                    *v *= 1.0 - h * h;
                }
            }
            Activation::LeakyReLU => {
                for (i, v) in g.data.iter_mut().enumerate() {
                    *v *= self.mask[i];
                }
            }
            Activation::None => {}
        }
        self.w1.backward(&g)
    }

    pub fn zero_grad(&mut self) {
        self.w1.zero_grad();
        self.w2.zero_grad();
    }

    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.w1.adam_step(lr, beta1, beta2, eps, weight_decay);
        self.w2.adam_step(lr, beta1, beta2, eps, weight_decay);
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let (w1, w2) = (&mut self.w1, &mut self.w2);
        vec![w1, w2]
    }
}

impl Layer for FeedForwardT {
    fn forward(&self, x: &Tensor) -> Tensor {
        FeedForwardT::forward(self, x)
    }

    fn forward_train(&mut self, x: &Matrix) -> Matrix {
        FeedForwardT::forward_train(self, x)
    }

    fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        FeedForwardT::backward(self, grad_out)
    }

    fn zero_grad(&mut self) {
        FeedForwardT::zero_grad(self);
    }

    fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        FeedForwardT::fa_update(self, grad_out, lr)
    }

    fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        FeedForwardT::adam_step(self, lr, beta1, beta2, eps, weight_decay);
    }

    fn parameters(&mut self) -> Vec<&mut LinearT> {
        FeedForwardT::parameters(self)
    }
}
