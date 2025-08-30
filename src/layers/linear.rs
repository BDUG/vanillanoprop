use super::layer::Layer;
use crate::math::Matrix;
use crate::rng::rng_from_env;
use crate::tensor::Tensor;
use rand::Rng;

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
    w_q: Vec<i8>,
    w_scale: f32,
    w_cached: bool,
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
        let rows = w.shape[0];
        let cols = w.shape[1];
        let grad = Matrix::zeros(rows, cols);
        let m = Matrix::zeros(rows, cols);
        let v = Matrix::zeros(rows, cols);
        // Pre-allocate with input dimensionality so most calls avoid resizing.
        let last_x = Matrix {
            rows: 0,
            cols: in_dim,
            data: Vec::with_capacity(in_dim),
        };
        let fb = Matrix::from_vec(
            out_dim,
            in_dim,
            (0..out_dim * in_dim)
                .map(|_| (rng.gen::<f32>() - 0.5) * 0.02)
                .collect(),
        );
        Self {
            w,
            grad,
            m,
            v,
            t: 0,
            last_x,
            fb,
            w_q: Vec::new(),
            w_scale: 0.0,
            w_cached: false,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::matmul(x, &self.w)
    }

    fn quantize_weights(&mut self) {
        let (w_q, w_scale) = self.w.quantize();
        self.w_q = w_q;
        self.w_scale = w_scale;
        self.w_cached = true;
    }

    /// Perform a quantized matrix multiplication using int8 operands.
    /// The input tensor is quantized on the fly. Quantized weights are cached
    /// so repeated calls avoid recomputing the quantization. When the
    /// `matrixmultiply` feature is enabled the multiply is delegated to the
    /// optimized `matrixmultiply` crate, otherwise a simple loop is used.
    pub fn quantized_matmul(&mut self, x: &Tensor) -> Tensor {
        let (x_q, x_scale) = x.quantize();
        if !self.w_cached {
            self.quantize_weights();
        }
        let rows = x.shape[0];
        let k = x.shape[1];
        let cols = self.w.shape[1];
        let mut out = vec![0f32; rows * cols];

        #[cfg(feature = "matrixmultiply")]
        unsafe {
            let x_f: Vec<f32> = x_q.iter().map(|&v| v as f32).collect();
            let w_f: Vec<f32> = self.w_q.iter().map(|&v| v as f32).collect();
            matrixmultiply::sgemm(
                rows,
                k,
                cols,
                1.0,
                x_f.as_ptr(),
                k as isize,
                1,
                w_f.as_ptr(),
                cols as isize,
                1,
                0.0,
                out.as_mut_ptr(),
                cols as isize,
                1,
            );
        }

        #[cfg(not(feature = "matrixmultiply"))]
        {
            for i in 0..rows {
                for j in 0..cols {
                    let mut sum = 0i32;
                    for p in 0..k {
                        let a = x_q[i * k + p] as i32;
                        let b = self.w_q[p * cols + j] as i32;
                        sum += a * b;
                    }
                    out[i * cols + j] = sum as f32;
                }
            }
        }

        let scale = x_scale * self.w_scale;
        for val in &mut out {
            *val /= scale;
        }
        Tensor::new(out, vec![rows, cols])
    }

    /// Forward pass storing the input for local/FA updates.
    pub fn forward_local(&mut self, x: &Matrix) -> Matrix {
        if self.last_x.rows != x.rows || self.last_x.cols != x.cols {
            self.last_x.rows = x.rows;
            self.last_x.cols = x.cols;
            self.last_x.data.resize(x.data.len(), 0.0);
        } else if self.last_x.data.len() != x.data.len() {
            self.last_x.data.resize(x.data.len(), 0.0);
        }
        self.last_x.data.clone_from_slice(&x.data);
        let w_m = Matrix::from_vec(self.w.shape[0], self.w.shape[1], self.w.data.clone());
        Matrix::matmul(&self.last_x, &w_m)
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
        for i in 0..self.w.data.len() {
            self.w.data[i] -= lr * grad_w.data[i];
        }
        self.w_cached = false;
        Matrix::matmul(grad_out, &self.fb)
    }

    /// Standard backward pass accumulating gradients (used for backprop)
    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let x_t = self.last_x.transpose();
        let grad_w = Matrix::matmul(&x_t, grad_out);
        self.grad = self.grad.add(&grad_w);
        let w_m = Matrix::from_vec(self.w.shape[0], self.w.shape[1], self.w.data.clone());
        Matrix::matmul(grad_out, &w_m.transpose())
    }

    pub fn zero_grad(&mut self) {
        self.grad = Matrix::zeros(self.grad.rows, self.grad.cols);
    }

    /// Adam optimisation step.  This is intentionally very small and only
    /// implements what is required for the training examples in this
    /// repository.
    pub fn sgd_step(&mut self, lr: f32, weight_decay: f32) {
        for i in 0..self.grad.data.len() {
            let g = self.grad.data[i] + weight_decay * self.w.data[i];
            self.w.data[i] -= lr * g;
        }
        self.w_cached = false;
    }

    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.t += 1;
        let beta1_t = beta1.powi(self.t as i32);
        let beta2_t = beta2.powi(self.t as i32);
        for i in 0..self.grad.data.len() {
            let g = self.grad.data[i] + weight_decay * self.w.data[i];
            self.m.data[i] = beta1 * self.m.data[i] + (1.0 - beta1) * g;
            self.v.data[i] = beta2 * self.v.data[i] + (1.0 - beta2) * g * g;
            let m_hat = self.m.data[i] / (1.0 - beta1_t);
            let v_hat = self.v.data[i] / (1.0 - beta2_t);
            self.w.data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        self.w_cached = false;
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

    fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        LinearT::adam_step(self, lr, beta1, beta2, eps, weight_decay);
    }

    fn parameters(&mut self) -> Vec<&mut LinearT> {
        LinearT::parameters(self)
    }
}
