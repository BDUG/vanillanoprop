use super::layer::Layer;
use super::linear::LinearT;
use crate::math::Matrix;
use crate::tensor::Tensor;

/// SwiGLU style gated feed-forward layer.
pub struct GatedFFNT {
    pub w_in: LinearT,
    pub w_out: LinearT,
    u: Matrix,
    v: Matrix,
    v_act: Matrix,
}

impl GatedFFNT {
    pub fn new(dim: usize, hidden: usize) -> Self {
        Self {
            w_in: LinearT::new(dim, hidden * 2),
            w_out: LinearT::new(hidden, dim),
            u: Matrix::zeros(0, 0),
            v: Matrix::zeros(0, 0),
            v_act: Matrix::zeros(0, 0),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.w_in.forward(x);
        let rows = h.shape[0];
        let cols = h.shape[1];
        let hidden = cols / 2;
        let mut gated = Matrix::zeros(rows, hidden);
        for r in 0..rows {
            for c in 0..hidden {
                let u = h.data[r * cols + c];
                let v = h.data[r * cols + hidden + c];
                let s = 1.0 / (1.0 + (-v).exp());
                let swish = v * s;
                gated.data[r * hidden + c] = u * swish;
            }
        }
        let g = Tensor::from_matrix(gated);
        self.w_out.forward(&g)
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        let h = self.w_in.forward_local(x);
        let rows = h.rows;
        let cols = h.cols;
        let hidden = cols / 2;
        self.u = Matrix::zeros(rows, hidden);
        self.v = Matrix::zeros(rows, hidden);
        self.v_act = Matrix::zeros(rows, hidden);
        let mut gated = Matrix::zeros(rows, hidden);
        for r in 0..rows {
            for c in 0..hidden {
                let u = h.data[r * cols + c];
                let v = h.data[r * cols + hidden + c];
                let s = 1.0 / (1.0 + (-v).exp());
                let swish = v * s;
                self.u.data[r * hidden + c] = u;
                self.v.data[r * hidden + c] = v;
                self.v_act.data[r * hidden + c] = swish;
                gated.data[r * hidden + c] = u * swish;
            }
        }
        self.w_out.forward_local(&gated)
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let grad_gated = self.w_out.backward(grad_out);
        let rows = grad_gated.rows;
        let hidden = grad_gated.cols;
        let mut grad_h = Matrix::zeros(rows, hidden * 2);
        for r in 0..rows {
            for c in 0..hidden {
                let g = grad_gated.data[r * hidden + c];
                let u = self.u.data[r * hidden + c];
                let v = self.v.data[r * hidden + c];
                let swish = self.v_act.data[r * hidden + c];
                grad_h.data[r * hidden * 2 + c] = g * swish;
                let s = 1.0 / (1.0 + (-v).exp());
                let swish_deriv = s + v * s * (1.0 - s);
                grad_h.data[r * hidden * 2 + hidden + c] = g * u * swish_deriv;
            }
        }
        self.w_in.backward(&grad_h)
    }

    pub fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        let grad_gated = self.w_out.fa_update(grad_out, lr);
        let rows = grad_gated.rows;
        let hidden = grad_gated.cols;
        let mut grad_h = Matrix::zeros(rows, hidden * 2);
        for r in 0..rows {
            for c in 0..hidden {
                let g = grad_gated.data[r * hidden + c];
                let u = self.u.data[r * hidden + c];
                let v = self.v.data[r * hidden + c];
                let swish = self.v_act.data[r * hidden + c];
                grad_h.data[r * hidden * 2 + c] = g * swish;
                let s = 1.0 / (1.0 + (-v).exp());
                let swish_deriv = s + v * s * (1.0 - s);
                grad_h.data[r * hidden * 2 + hidden + c] = g * u * swish_deriv;
            }
        }
        self.w_in.fa_update(&grad_h, lr)
    }

    pub fn zero_grad(&mut self) {
        self.w_in.zero_grad();
        self.w_out.zero_grad();
    }

    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.w_in.adam_step(lr, beta1, beta2, eps, weight_decay);
        self.w_out.adam_step(lr, beta1, beta2, eps, weight_decay);
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let (w1, w2) = (&mut self.w_in, &mut self.w_out);
        vec![w1, w2]
    }
}

impl Layer for GatedFFNT {
    fn forward(&self, x: &Tensor) -> Tensor {
        GatedFFNT::forward(self, x)
    }

    fn forward_train(&mut self, x: &Matrix) -> Matrix {
        GatedFFNT::forward_train(self, x)
    }

    fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        GatedFFNT::backward(self, grad_out)
    }

    fn zero_grad(&mut self) {
        GatedFFNT::zero_grad(self);
    }

    fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        GatedFFNT::fa_update(self, grad_out, lr)
    }

    fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        GatedFFNT::adam_step(self, lr, beta1, beta2, eps, weight_decay);
    }

    fn parameters(&mut self) -> Vec<&mut LinearT> {
        GatedFFNT::parameters(self)
    }
}

