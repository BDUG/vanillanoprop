use crate::tensor::Tensor;
use crate::math::Matrix;
use super::linear::LinearT;
use super::layer::Layer;

pub struct MultiHeadAttentionT {
    pub wq: LinearT,
    pub wk: LinearT,
    pub wv: LinearT,
    pub wo: LinearT,
    // caches for backward
    x: Matrix,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    attn: Matrix,
    scores: Matrix,
}

impl MultiHeadAttentionT {
    pub fn new(model_dim: usize) -> Self {
        Self {
            wq: LinearT::new(model_dim, model_dim),
            wk: LinearT::new(model_dim, model_dim),
            wv: LinearT::new(model_dim, model_dim),
            wo: LinearT::new(model_dim, model_dim),
            x: Matrix::zeros(0, 0),
            q: Matrix::zeros(0, 0),
            k: Matrix::zeros(0, 0),
            v: Matrix::zeros(0, 0),
            attn: Matrix::zeros(0, 0),
            scores: Matrix::zeros(0, 0),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);

        // ⚠️ stark vereinfachte Attention: keine Aufteilung in Köpfe,
        // kein Softmax, kein Masking – nur MatMul zur Demo
        let kt = Tensor::transpose(&k);
        let attn = Tensor::matmul(&q, &kt);
        let scores = Tensor::matmul(&attn, &v);
        self.wo.forward(&scores)
    }

    pub fn forward_local(&mut self, x: &Matrix) -> Matrix {
        self.x = x.clone();
        self.q = self.wq.forward_local(x);
        self.k = self.wk.forward_local(x);
        self.v = self.wv.forward_local(x);
        let kt = self.k.transpose();
        self.attn = Matrix::matmul(&self.q, &kt);
        self.scores = Matrix::matmul(&self.attn, &self.v);
        self.wo.forward_local(&self.scores)
    }

    pub fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        let grad_scores = self.wo.fa_update(grad_out, lr);
        let grad_attn = Matrix::matmul(&grad_scores, &self.v.transpose());
        let grad_v = Matrix::matmul(&self.attn.transpose(), &grad_scores);
        let grad_q = Matrix::matmul(&grad_attn, &self.k);
        let grad_k = Matrix::matmul(&grad_attn.transpose(), &self.q);
        let gx_q = self.wq.fa_update(&grad_q, lr);
        let gx_k = self.wk.fa_update(&grad_k, lr);
        let gx_v = self.wv.fa_update(&grad_v, lr);
        let tmp = Matrix::add(&gx_q, &gx_k);
        Matrix::add(&tmp, &gx_v)
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        self.forward_local(x)
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let grad_scores = self.wo.backward(grad_out);
        let grad_attn = Matrix::matmul(&grad_scores, &self.v.transpose());
        let grad_v = Matrix::matmul(&self.attn.transpose(), &grad_scores);
        let grad_q = Matrix::matmul(&grad_attn, &self.k);
        let grad_k = Matrix::matmul(&grad_attn.transpose(), &self.q);
        let gx_q = self.wq.backward(&grad_q);
        let gx_k = self.wk.backward(&grad_k);
        let gx_v = self.wv.backward(&grad_v);
        let tmp = Matrix::add(&gx_q, &gx_k);
        Matrix::add(&tmp, &gx_v)
    }

    pub fn zero_grad(&mut self) {
        self.wq.zero_grad();
        self.wk.zero_grad();
        self.wv.zero_grad();
        self.wo.zero_grad();
    }

    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.wq
            .adam_step(lr, beta1, beta2, eps, weight_decay);
        self.wk
            .adam_step(lr, beta1, beta2, eps, weight_decay);
        self.wv
            .adam_step(lr, beta1, beta2, eps, weight_decay);
        self.wo
            .adam_step(lr, beta1, beta2, eps, weight_decay);
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let (wq, wk, wv, wo) = (&mut self.wq, &mut self.wk, &mut self.wv, &mut self.wo);
       vec![wq, wk, wv, wo]
    }
}

impl Layer for MultiHeadAttentionT {
    fn forward(&self, x: &Tensor) -> Tensor {
        MultiHeadAttentionT::forward(self, x)
    }

    fn forward_train(&mut self, x: &Matrix) -> Matrix {
        MultiHeadAttentionT::forward_train(self, x)
    }

    fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        MultiHeadAttentionT::backward(self, grad_out)
    }

    fn zero_grad(&mut self) {
        MultiHeadAttentionT::zero_grad(self);
    }

    fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        MultiHeadAttentionT::fa_update(self, grad_out, lr)
    }

    fn adam_step(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) {
        MultiHeadAttentionT::adam_step(self, lr, beta1, beta2, eps, weight_decay);
    }

    fn parameters(&mut self) -> Vec<&mut LinearT> {
        MultiHeadAttentionT::parameters(self)
    }
}

