use crate::tensor::Tensor;
use crate::math::Matrix;
use super::layer::Layer;
use rand::Rng;
use crate::rng::rng_from_env;

// Simple linear module with rudimentary autograd support.  During training
// each `LinearT` stores the last input that was seen so that a backward pass
// can compute gradients for both the input and the weight matrix.  In
// addition, the struct keeps Adam optimizer statistics so that the optimizer
// state automatically persists across iterations.

pub struct LinearT {
    pub w: Tensor,
    grad: Matrix,
    m: Matrix,
    v: Matrix,
    t: usize,
    last_x: Matrix,
    fb: Matrix,
}

impl LinearT {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let mut rng = rng_from_env();
        let data = Matrix::from_vec(
            in_dim,
            out_dim,
            (0..in_dim * out_dim)
                .map(|_| (rng.gen::<f32>() - 0.5) * 0.02)
                .collect(),
        );
        let w = Tensor::from_matrix(data);
        let grad = Matrix::zeros(w.data.rows, w.data.cols);
        let m = Matrix::zeros(w.data.rows, w.data.cols);
        let v = Matrix::zeros(w.data.rows, w.data.cols);
        let last_x = Matrix::zeros(0, 0);
        let fb = Matrix::from_vec(
            out_dim,
            in_dim,
            (0..out_dim * in_dim)
                .map(|_| (rng.gen::<f32>() - 0.5) * 0.02)
                .collect(),
        );
        Self { w, grad, m, v, t: 0, last_x, fb }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::matmul(x, &self.w)
    }

    /// Forward pass storing the input for local/FA updates.
    pub fn forward_local(&mut self, x: &Matrix) -> Matrix {
        self.last_x = x.clone();
        Matrix::matmul(x, &self.w.data)
    }

    /// Backward-compatible training forward used by backprop examples.
    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        self.forward_local(x)
    }

    /// Feedback alignment style local update.  `grad_out` is the error signal
    /// for the layer output.  The weights are updated using a Hebbian rule and
    /// a fixed random feedback matrix is used to propagate the error to the
    /// previous layer.
    pub fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        let x_t = self.last_x.transpose();
        let grad_w = Matrix::matmul(&x_t, grad_out);
        for i in 0..self.w.data.data.len() {
            self.w.data.data[i] -= lr * grad_w.data[i];
        }
        Matrix::matmul(grad_out, &self.fb)
    }

    /// Standard backward pass accumulating gradients (used for backprop)
    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let x_t = self.last_x.transpose();
        let grad_w = Matrix::matmul(&x_t, grad_out);
        self.grad = self.grad.add(&grad_w);
        Matrix::matmul(grad_out, &self.w.data.transpose())
    }

    pub fn zero_grad(&mut self) {
        self.grad = Matrix::zeros(self.grad.rows, self.grad.cols);
    }

    /// Adam optimisation step.  This is intentionally very small and only
    /// implements what is required for the training examples in this
    /// repository.
    pub fn sgd_step(&mut self, lr: f32, weight_decay: f32) {
        for i in 0..self.grad.data.len() {
            let g = self.grad.data[i] + weight_decay * self.w.data.data[i];
            self.w.data.data[i] -= lr * g;
        }
    }

    pub fn adam_step(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) {
        self.t += 1;
        for i in 0..self.grad.data.len() {
            let g = self.grad.data[i] + weight_decay * self.w.data.data[i];
            self.m.data[i] = beta1 * self.m.data[i] + (1.0 - beta1) * g;
            self.v.data[i] = beta2 * self.v.data[i] + (1.0 - beta2) * g * g;
            let m_hat = self.m.data[i] / (1.0 - beta1.powi(self.t as i32));
            let v_hat = self.v.data[i] / (1.0 - beta2.powi(self.t as i32));
            self.w.data.data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        vec![self]
    }
}

impl Layer for LinearT {
    fn forward(&self, x: &Tensor) -> Tensor {
        LinearT::forward(self, x)
    }

    fn forward_train(&mut self, x: &Matrix) -> Matrix {
        LinearT::forward_train(self, x)
    }

    fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        LinearT::backward(self, grad_out)
    }

    fn zero_grad(&mut self) {
        LinearT::zero_grad(self);
    }

    fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        LinearT::fa_update(self, grad_out, lr)
    }

    fn adam_step(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) {
        LinearT::adam_step(self, lr, beta1, beta2, eps, weight_decay);
    }

    fn parameters(&mut self) -> Vec<&mut LinearT> {
        LinearT::parameters(self)
    }
}

