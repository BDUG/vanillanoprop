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

    fn adam_step(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        t: usize,
        weight_decay: f32,
    ) {
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

/// Batch normalization layer with learnable scale (`gamma`) and shift (`beta`).
pub struct BatchNorm {
    pub gamma: Param,
    pub beta: Param,
    eps: f32,
    momentum: f32,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
    x_hat: Matrix,
    mean: Vec<f32>,
    var: Vec<f32>,
    t: usize,
}

impl BatchNorm {
    pub fn new(dim: usize, eps: f32, momentum: f32) -> Self {
        Self {
            gamma: Param::new(dim, 1.0),
            beta: Param::new(dim, 0.0),
            eps,
            momentum,
            running_mean: vec![0.0; dim],
            running_var: vec![1.0; dim],
            x_hat: Matrix::zeros(0, 0),
            mean: vec![0.0; dim],
            var: vec![1.0; dim],
            t: 0,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let rows = x.data.rows;
        let cols = x.data.cols;
        let mut out = Matrix::zeros(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let x_hat = (x.data.data[idx] - self.running_mean[c])
                    / (self.running_var[c] + self.eps).sqrt();
                out.data[idx] = self.gamma.w[c] * x_hat + self.beta.w[c];
            }
        }
        Tensor::from_matrix(out)
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        let rows = x.rows;
        let cols = x.cols;
        let n = rows as f32;
        self.mean = vec![0.0; cols];
        self.var = vec![0.0; cols];
        for c in 0..cols {
            let mut sum = 0.0;
            for r in 0..rows {
                sum += x.data[r * cols + c];
            }
            let m = sum / n;
            self.mean[c] = m;
            let mut var_sum = 0.0;
            for r in 0..rows {
                let val = x.data[r * cols + c] - m;
                var_sum += val * val;
            }
            let v = var_sum / n;
            self.var[c] = v;
            self.running_mean[c] = self.momentum * self.running_mean[c] + (1.0 - self.momentum) * m;
            self.running_var[c] = self.momentum * self.running_var[c] + (1.0 - self.momentum) * v;
        }
        self.x_hat = Matrix::zeros(rows, cols);
        let mut out = Matrix::zeros(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let x_hat = (x.data[idx] - self.mean[c]) / (self.var[c] + self.eps).sqrt();
                self.x_hat.data[idx] = x_hat;
                out.data[idx] = self.gamma.w[c] * x_hat + self.beta.w[c];
            }
        }
        out
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let rows = grad_out.rows;
        let cols = grad_out.cols;
        let n = rows as f32;
        let mut grad_input = Matrix::zeros(rows, cols);
        for c in 0..cols {
            let mut sum_dy = 0.0;
            let mut sum_dy_xhat = 0.0;
            let mut sum_dxhat = 0.0;
            let mut sum_dxhat_xhat = 0.0;
            for r in 0..rows {
                let idx = r * cols + c;
                let dy = grad_out.data[idx];
                sum_dy += dy;
                sum_dy_xhat += dy * self.x_hat.data[idx];
                let dxhat = dy * self.gamma.w[c];
                sum_dxhat += dxhat;
                sum_dxhat_xhat += dxhat * self.x_hat.data[idx];
            }
            self.beta.grad[c] += sum_dy;
            self.gamma.grad[c] += sum_dy_xhat;
            let inv_std = 1.0 / (self.var[c] + self.eps).sqrt();
            for r in 0..rows {
                let idx = r * cols + c;
                let dxhat = grad_out.data[idx] * self.gamma.w[c];
                grad_input.data[idx] =
                    (dxhat * n - sum_dxhat - self.x_hat.data[idx] * sum_dxhat_xhat) * inv_std / n;
            }
        }
        grad_input
    }

    pub fn zero_grad(&mut self) {
        self.gamma.zero_grad();
        self.beta.zero_grad();
    }

    pub fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        let grad_input = self.backward(grad_out);
        self.gamma.sgd_step(lr, 0.0);
        self.beta.sgd_step(lr, 0.0);
        self.zero_grad();
        grad_input
    }

    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.t += 1;
        self.gamma
            .adam_step(lr, beta1, beta2, eps, self.t, weight_decay);
        self.beta
            .adam_step(lr, beta1, beta2, eps, self.t, weight_decay);
    }

    pub fn parameters(&mut self) -> Vec<&mut crate::layers::linear::LinearT> {
        Vec::new()
    }
}

impl Layer for BatchNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        BatchNorm::forward(self, x)
    }

    fn forward_train(&mut self, x: &Matrix) -> Matrix {
        BatchNorm::forward_train(self, x)
    }

    fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        BatchNorm::backward(self, grad_out)
    }

    fn zero_grad(&mut self) {
        BatchNorm::zero_grad(self);
    }

    fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        BatchNorm::fa_update(self, grad_out, lr)
    }

    fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        BatchNorm::adam_step(self, lr, beta1, beta2, eps, weight_decay);
    }

    fn parameters(&mut self) -> Vec<&mut crate::layers::linear::LinearT> {
        BatchNorm::parameters(self)
    }
}

/// Layer normalization with learnable scale (`gamma`) and shift (`beta`).
pub struct LayerNorm {
    pub gamma: Param,
    pub beta: Param,
    eps: f32,
    x_hat: Matrix,
    inv_std: Vec<f32>,
    t: usize,
}

impl LayerNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            gamma: Param::new(dim, 1.0),
            beta: Param::new(dim, 0.0),
            eps,
            x_hat: Matrix::zeros(0, 0),
            inv_std: Vec::new(),
            t: 0,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let rows = x.data.rows;
        let cols = x.data.cols;
        let mut out = Matrix::zeros(rows, cols);
        for r in 0..rows {
            let mut mean = 0.0;
            for c in 0..cols {
                mean += x.data.data[r * cols + c];
            }
            mean /= cols as f32;
            let mut var = 0.0;
            for c in 0..cols {
                let v = x.data.data[r * cols + c] - mean;
                var += v * v;
            }
            var /= cols as f32;
            let inv_std = 1.0 / (var + self.eps).sqrt();
            for c in 0..cols {
                let idx = r * cols + c;
                let x_hat = (x.data.data[idx] - mean) * inv_std;
                out.data[idx] = self.gamma.w[c] * x_hat + self.beta.w[c];
            }
        }
        Tensor::from_matrix(out)
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        let rows = x.rows;
        let cols = x.cols;
        self.x_hat = Matrix::zeros(rows, cols);
        self.inv_std = vec![0.0; rows];
        let mut out = Matrix::zeros(rows, cols);
        for r in 0..rows {
            let mut mean = 0.0;
            for c in 0..cols {
                mean += x.data[r * cols + c];
            }
            mean /= cols as f32;
            let mut var = 0.0;
            for c in 0..cols {
                let idx = r * cols + c;
                let diff = x.data[idx] - mean;
                var += diff * diff;
            }
            var /= cols as f32;
            let inv_std = 1.0 / (var + self.eps).sqrt();
            self.inv_std[r] = inv_std;
            for c in 0..cols {
                let idx = r * cols + c;
                let x_hat = (x.data[idx] - mean) * inv_std;
                self.x_hat.data[idx] = x_hat;
                out.data[idx] = self.gamma.w[c] * x_hat + self.beta.w[c];
            }
        }
        out
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let rows = grad_out.rows;
        let cols = grad_out.cols;
        let n = cols as f32;
        let mut grad_input = Matrix::zeros(rows, cols);
        for c in 0..cols {
            let mut sum_dy = 0.0;
            let mut sum_dy_xhat = 0.0;
            for r in 0..rows {
                let idx = r * cols + c;
                let dy = grad_out.data[idx];
                sum_dy += dy;
                sum_dy_xhat += dy * self.x_hat.data[idx];
            }
            self.beta.grad[c] += sum_dy;
            self.gamma.grad[c] += sum_dy_xhat;
        }
        for r in 0..rows {
            let mut sum_dxhat = 0.0;
            let mut sum_dxhat_xhat = 0.0;
            for c in 0..cols {
                let idx = r * cols + c;
                let dxhat = grad_out.data[idx] * self.gamma.w[c];
                sum_dxhat += dxhat;
                sum_dxhat_xhat += dxhat * self.x_hat.data[idx];
            }
            for c in 0..cols {
                let idx = r * cols + c;
                let dxhat = grad_out.data[idx] * self.gamma.w[c];
                grad_input.data[idx] = self.inv_std[r]
                    * (dxhat - sum_dxhat / n - self.x_hat.data[idx] * sum_dxhat_xhat / n);
            }
        }
        grad_input
    }

    pub fn zero_grad(&mut self) {
        self.gamma.zero_grad();
        self.beta.zero_grad();
    }

    pub fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        let grad_input = self.backward(grad_out);
        self.gamma.sgd_step(lr, 0.0);
        self.beta.sgd_step(lr, 0.0);
        self.zero_grad();
        grad_input
    }

    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.t += 1;
        self.gamma
            .adam_step(lr, beta1, beta2, eps, self.t, weight_decay);
        self.beta
            .adam_step(lr, beta1, beta2, eps, self.t, weight_decay);
    }

    pub fn parameters(&mut self) -> Vec<&mut crate::layers::linear::LinearT> {
        Vec::new()
    }
}

impl Layer for LayerNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        LayerNorm::forward(self, x)
    }

    fn forward_train(&mut self, x: &Matrix) -> Matrix {
        LayerNorm::forward_train(self, x)
    }

    fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        LayerNorm::backward(self, grad_out)
    }

    fn zero_grad(&mut self) {
        LayerNorm::zero_grad(self);
    }

    fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        LayerNorm::fa_update(self, grad_out, lr)
    }

    fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        LayerNorm::adam_step(self, lr, beta1, beta2, eps, weight_decay);
    }

    fn parameters(&mut self) -> Vec<&mut crate::layers::linear::LinearT> {
        LayerNorm::parameters(self)
    }
}
