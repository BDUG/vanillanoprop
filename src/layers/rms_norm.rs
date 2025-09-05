use super::layer::Layer;
use crate::math::Matrix;
use crate::tensor::Tensor;

/// Simple 1D parameter with Adam optimiser statistics.
pub struct Param {
    pub w: Vec<f32>,
    grad: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl Param {
    fn new(dim: usize, init: f32) -> Self {
        Self {
            w: vec![init; dim],
            grad: vec![0.0; dim],
            m: vec![0.0; dim],
            v: vec![0.0; dim],
        }
    }

    fn zero_grad(&mut self) {
        for g in self.grad.iter_mut() {
            *g = 0.0;
        }
    }

    fn sgd_step(&mut self, lr: f32, weight_decay: f32) {
        for i in 0..self.w.len() {
            let g = self.grad[i] + weight_decay * self.w[i];
            self.w[i] -= lr * g;
        }
    }

    fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, t: usize, weight_decay: f32) {
        for i in 0..self.w.len() {
            let g = self.grad[i] + weight_decay * self.w[i];
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;
            let m_hat = self.m[i] / (1.0 - beta1.powi(t as i32));
            let v_hat = self.v[i] / (1.0 - beta2.powi(t as i32));
            self.w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

/// Root mean square normalisation layer with learnable scale parameter.
pub struct RmsNorm {
    pub gamma: Param,
    eps: f32,
    x: Matrix,
    inv_rms: Vec<f32>,
    t: usize,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            gamma: Param::new(dim, 1.0),
            eps,
            x: Matrix::zeros(0, 0),
            inv_rms: Vec::new(),
            t: 0,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let rows = x.shape[0];
        let cols = x.shape[1];
        let mut out = Matrix::zeros(rows, cols);
        for r in 0..rows {
            let mut sum_sq = 0.0;
            for c in 0..cols {
                let v = x.data[r * cols + c];
                sum_sq += v * v;
            }
            let inv_rms = 1.0 / ((sum_sq / cols as f32) + self.eps).sqrt();
            for c in 0..cols {
                let idx = r * cols + c;
                out.data[idx] = x.data[idx] * self.gamma.w[c] * inv_rms;
            }
        }
        Tensor::from_matrix(out)
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        let rows = x.rows;
        let cols = x.cols;
        self.x = Matrix::from_vec(rows, cols, x.data.clone());
        self.inv_rms = vec![0.0; rows];
        let mut out = Matrix::zeros(rows, cols);
        for r in 0..rows {
            let mut sum_sq = 0.0;
            for c in 0..cols {
                let v = x.data[r * cols + c];
                sum_sq += v * v;
            }
            let inv_rms = 1.0 / ((sum_sq / cols as f32) + self.eps).sqrt();
            self.inv_rms[r] = inv_rms;
            for c in 0..cols {
                let idx = r * cols + c;
                out.data[idx] = x.data[idx] * self.gamma.w[c] * inv_rms;
            }
        }
        out
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let rows = grad_out.rows;
        let cols = grad_out.cols;
        let n = cols as f32;
        let mut grad_input = Matrix::zeros(rows, cols);
        for r in 0..rows {
            let inv_rms = self.inv_rms[r];
            let mut dot = 0.0;
            for c in 0..cols {
                let idx = r * cols + c;
                let x_val = self.x.data[idx];
                let go = grad_out.data[idx];
                dot += self.gamma.w[c] * x_val * go;
                self.gamma.grad[c] += go * x_val * inv_rms;
            }
            let inv_rms3 = inv_rms * inv_rms * inv_rms;
            for c in 0..cols {
                let idx = r * cols + c;
                let x_val = self.x.data[idx];
                let go = grad_out.data[idx];
                grad_input.data[idx] = inv_rms * self.gamma.w[c] * go
                    - inv_rms3 * x_val * dot / n;
            }
        }
        grad_input
    }

    pub fn zero_grad(&mut self) {
        self.gamma.zero_grad();
    }

    pub fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        let grad_input = self.backward(grad_out);
        self.gamma.sgd_step(lr, 0.0);
        self.zero_grad();
        grad_input
    }

    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.t += 1;
        self.gamma
            .adam_step(lr, beta1, beta2, eps, self.t, weight_decay);
    }

    pub fn parameters(&mut self) -> Vec<&mut crate::layers::linear::LinearT> {
        Vec::new()
    }
}

impl Layer for RmsNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        RmsNorm::forward(self, x)
    }

    fn forward_train(&mut self, x: &Matrix) -> Matrix {
        RmsNorm::forward_train(self, x)
    }

    fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        RmsNorm::backward(self, grad_out)
    }

    fn zero_grad(&mut self) {
        RmsNorm::zero_grad(self);
    }

    fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        RmsNorm::fa_update(self, grad_out, lr)
    }

    fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        RmsNorm::adam_step(self, lr, beta1, beta2, eps, weight_decay);
    }

    fn parameters(&mut self) -> Vec<&mut crate::layers::linear::LinearT> {
        RmsNorm::parameters(self)
    }
}

