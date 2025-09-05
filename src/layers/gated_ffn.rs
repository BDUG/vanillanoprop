use super::layer::Layer;
use super::linear::LinearT;
use crate::math::Matrix;
use crate::tensor::Tensor;

/// SwiGLU style gated feed-forward network.
pub struct GatedFFN {
    pub w1: LinearT,
    pub w2: LinearT,
    // caches for backward
    u: Matrix,
    v: Matrix,
    h: Matrix,
}

impl GatedFFN {
    pub fn new(dim: usize, hidden: usize) -> Self {
        Self {
            w1: LinearT::new(dim, hidden * 2),
            w2: LinearT::new(hidden, dim),
            u: Matrix::zeros(0, 0),
            v: Matrix::zeros(0, 0),
            h: Matrix::zeros(0, 0),
        }
    }

    fn swish(x: f32) -> f32 {
        let sig = 1.0 / (1.0 + (-x).exp());
        x * sig
    }

    fn swish_grad(x: f32) -> f32 {
        let sig = 1.0 / (1.0 + (-x).exp());
        sig + x * sig * (1.0 - sig)
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.w1.forward(x);
        let rows = h.shape[0];
        let hidden = self.w2.w.shape[0];
        let mut act = vec![0.0; rows * hidden];
        for r in 0..rows {
            for c in 0..hidden {
                let u = h.data[r * hidden * 2 + c];
                let v = h.data[r * hidden * 2 + c + hidden];
                let sw = GatedFFN::swish(u);
                act[r * hidden + c] = sw * v;
            }
        }
        let g = Tensor::new(act, vec![rows, hidden]);
        self.w2.forward(&g)
    }

    fn forward_local_inner(&mut self, x: &Matrix) -> Matrix {
        let tmp = self.w1.forward_local(x);
        let rows = tmp.rows;
        let hidden = self.w2.w.shape[0];
        self.u = Matrix::zeros(rows, hidden);
        self.v = Matrix::zeros(rows, hidden);
        self.h = Matrix::zeros(rows, hidden);
        for r in 0..rows {
            for c in 0..hidden {
                let u = tmp.data[r * hidden * 2 + c];
                let v = tmp.data[r * hidden * 2 + c + hidden];
                let sw = GatedFFN::swish(u);
                self.u.data[r * hidden + c] = u;
                self.v.data[r * hidden + c] = v;
                self.h.data[r * hidden + c] = sw * v;
            }
        }
        let out = self.w2.forward_local(&self.h);
        out
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        self.forward_local_inner(x)
    }

    pub fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        let grad_h = self.w2.fa_update(grad_out, lr);
        let rows = grad_h.rows;
        let hidden = grad_h.cols;
        let mut grad_tmp = Matrix::zeros(rows, hidden * 2);
        for r in 0..rows {
            for c in 0..hidden {
                let idx = r * hidden + c;
                let grad = grad_h.data[idx];
                let u = self.u.data[idx];
                let v = self.v.data[idx];
                let sw = GatedFFN::swish(u);
                grad_tmp.data[r * hidden * 2 + c + hidden] = grad * sw;
                let sw_grad = GatedFFN::swish_grad(u);
                grad_tmp.data[r * hidden * 2 + c] = grad * v * sw_grad;
            }
        }
        self.w1.fa_update(&grad_tmp, lr)
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let grad_h = self.w2.backward(grad_out);
        let rows = grad_h.rows;
        let hidden = grad_h.cols;
        let mut grad_tmp = Matrix::zeros(rows, hidden * 2);
        for r in 0..rows {
            for c in 0..hidden {
                let idx = r * hidden + c;
                let grad = grad_h.data[idx];
                let u = self.u.data[idx];
                let v = self.v.data[idx];
                let sw = GatedFFN::swish(u);
                grad_tmp.data[r * hidden * 2 + c + hidden] = grad * sw;
                let sw_grad = GatedFFN::swish_grad(u);
                grad_tmp.data[r * hidden * 2 + c] = grad * v * sw_grad;
            }
        }
        self.w1.backward(&grad_tmp)
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

impl Layer for GatedFFN {
    fn forward(&self, x: &Tensor) -> Tensor {
        GatedFFN::forward(self, x)
    }

    fn forward_train(&mut self, x: &Matrix) -> Matrix {
        GatedFFN::forward_train(self, x)
    }

    fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        GatedFFN::backward(self, grad_out)
    }

    fn zero_grad(&mut self) {
        GatedFFN::zero_grad(self);
    }

    fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        GatedFFN::fa_update(self, grad_out, lr)
    }

    fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        GatedFFN::adam_step(self, lr, beta1, beta2, eps, weight_decay);
    }

    fn parameters(&mut self) -> Vec<&mut LinearT> {
        GatedFFN::parameters(self)
    }
}
