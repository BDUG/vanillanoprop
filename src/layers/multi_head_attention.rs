use super::layer::Layer;
use super::linear::LinearT;
use crate::math::{Matrix, MatrixLike};
use crate::tensor::Tensor;

fn copy_into(dest: &mut Matrix, start_col: usize, src: &Matrix) {
    for r in 0..dest.rows {
        let dst = &mut dest.data[r * dest.cols + start_col..r * dest.cols + start_col + src.cols];
        let s = &src.data[r * src.cols..(r + 1) * src.cols];
        dst.copy_from_slice(s);
    }
}

fn softmax_backward<A: MatrixLike, B: MatrixLike>(attn: &A, grad_out: &B) -> Matrix {
    let mut grad = Matrix::zeros(grad_out.rows(), grad_out.cols());
    for r in 0..grad_out.rows() {
        let mut dot = 0.0;
        for c in 0..grad_out.cols() {
            dot += grad_out.get(r, c) * attn.get(r, c);
        }
        for c in 0..grad_out.cols() {
            let v = attn.get(r, c) * (grad_out.get(r, c) - dot);
            grad.set(r, c, v);
        }
    }
    grad
}

pub struct MultiHeadAttentionT {
    pub wq: LinearT,
    pub wk: LinearT,
    pub wv: LinearT,
    pub wo: LinearT,
    pub num_heads: usize,
    // caches for backward
    x: Matrix,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    attn: Matrix,
    scores: Matrix,
    mask: Option<Matrix>,
}

impl MultiHeadAttentionT {
    pub fn new(model_dim: usize, num_heads: usize) -> Self {
        assert!(model_dim % num_heads == 0);
        Self {
            wq: LinearT::new(model_dim, model_dim),
            wk: LinearT::new(model_dim, model_dim),
            wv: LinearT::new(model_dim, model_dim),
            wo: LinearT::new(model_dim, model_dim),
            num_heads,
            x: Matrix::zeros(0, 0),
            q: Matrix::zeros(0, 0),
            k: Matrix::zeros(0, 0),
            v: Matrix::zeros(0, 0),
            attn: Matrix::zeros(0, 0),
            scores: Matrix::zeros(0, 0),
            mask: None,
        }
    }

    pub fn set_mask(&mut self, mask: Matrix) {
        self.mask = Some(mask);
    }

    pub fn clear_mask(&mut self) {
        self.mask = None;
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let q_t = self.wq.forward(x);
        let k_t = self.wk.forward(x);
        let v_t = self.wv.forward(x);
        let q = Matrix::from_vec(q_t.shape[0], q_t.shape[1], q_t.data.clone());
        let k = Matrix::from_vec(k_t.shape[0], k_t.shape[1], k_t.data.clone());
        let v = Matrix::from_vec(v_t.shape[0], v_t.shape[1], v_t.data.clone());
        let seq_len = q.rows;
        let model_dim = q.cols;
        let head_dim = model_dim / self.num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut concat = Matrix::zeros(seq_len, model_dim);
        for h in 0..self.num_heads {
            let qh = q.view_cols(h * head_dim, head_dim);
            let kh = k.view_cols(h * head_dim, head_dim);
            let vh = v.view_cols(h * head_dim, head_dim);
            let mut scores = Matrix::matmul_views(&qh, &kh.transpose());
            for s in scores.data.iter_mut() {
                *s *= scale;
            }
            if let Some(m) = self.mask.as_ref() {
                for i in 0..scores.data.len() {
                    scores.data[i] += m.data[i];
                }
            }
            let attn = scores.softmax();
            let head_out = Matrix::matmul_views(&attn, &vh);
            copy_into(&mut concat, h * head_dim, &head_out);
        }
        let t = Tensor::from_matrix(concat);
        self.wo.forward(&t)
    }

    pub fn forward_local(&mut self, x: &Matrix) -> Matrix {
        self.x = x.clone();
        self.q = self.wq.forward_local(x);
        self.k = self.wk.forward_local(x);
        self.v = self.wv.forward_local(x);
        let seq_len = self.q.rows;
        let model_dim = self.q.cols;
        let head_dim = model_dim / self.num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        self.attn = Matrix::zeros(self.num_heads * seq_len, seq_len);
        self.scores = Matrix::zeros(seq_len, model_dim);
        for h in 0..self.num_heads {
            let qh = self.q.view_cols(h * head_dim, head_dim);
            let kh = self.k.view_cols(h * head_dim, head_dim);
            let vh = self.v.view_cols(h * head_dim, head_dim);
            let mut scores = Matrix::matmul_views(&qh, &kh.transpose());
            for s in scores.data.iter_mut() {
                *s *= scale;
            }
            if let Some(m) = self.mask.as_ref() {
                for i in 0..scores.data.len() {
                    scores.data[i] += m.data[i];
                }
            }
            let attn = scores.softmax();
            for r in 0..seq_len {
                let dst = &mut self.attn.data
                    [(h * seq_len + r) * seq_len..(h * seq_len + r + 1) * seq_len];
                let src = &attn.data[r * seq_len..(r + 1) * seq_len];
                dst.copy_from_slice(src);
            }
            let head_out = Matrix::matmul_views(&attn, &vh);
            copy_into(&mut self.scores, h * head_dim, &head_out);
        }
        self.wo.forward_local(&self.scores)
    }

    pub fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        let grad_concat = self.wo.fa_update(grad_out, lr);
        let seq_len = self.q.rows;
        let model_dim = self.q.cols;
        let head_dim = model_dim / self.num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut grad_q = Matrix::zeros(seq_len, model_dim);
        let mut grad_k = Matrix::zeros(seq_len, model_dim);
        let mut grad_v = Matrix::zeros(seq_len, model_dim);
        for h in 0..self.num_heads {
            let qh = self.q.view_cols(h * head_dim, head_dim);
            let kh = self.k.view_cols(h * head_dim, head_dim);
            let vh = self.v.view_cols(h * head_dim, head_dim);
            let attn = self.attn.view_rows(h * seq_len, seq_len);
            let grad_head = grad_concat.view_cols(h * head_dim, head_dim);
            let grad_vh = Matrix::matmul_views(&attn.transpose(), &grad_head);
            copy_into(&mut grad_v, h * head_dim, &grad_vh);
            let grad_attn = Matrix::matmul_views(&grad_head, &vh.transpose());
            let mut grad_scores = softmax_backward(&attn, &grad_attn);
            for s in grad_scores.data.iter_mut() {
                *s *= scale;
            }
            let grad_qh = Matrix::matmul_views(&grad_scores, &kh);
            let grad_kh = Matrix::matmul_views(&grad_scores.transpose(), &qh);
            copy_into(&mut grad_q, h * head_dim, &grad_qh);
            copy_into(&mut grad_k, h * head_dim, &grad_kh);
        }
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
        let grad_concat = self.wo.backward(grad_out);
        let seq_len = self.q.rows;
        let model_dim = self.q.cols;
        let head_dim = model_dim / self.num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut grad_q = Matrix::zeros(seq_len, model_dim);
        let mut grad_k = Matrix::zeros(seq_len, model_dim);
        let mut grad_v = Matrix::zeros(seq_len, model_dim);
        for h in 0..self.num_heads {
            let qh = self.q.view_cols(h * head_dim, head_dim);
            let kh = self.k.view_cols(h * head_dim, head_dim);
            let vh = self.v.view_cols(h * head_dim, head_dim);
            let attn = self.attn.view_rows(h * seq_len, seq_len);
            let grad_head = grad_concat.view_cols(h * head_dim, head_dim);
            let grad_vh = Matrix::matmul_views(&attn.transpose(), &grad_head);
            copy_into(&mut grad_v, h * head_dim, &grad_vh);
            let grad_attn = Matrix::matmul_views(&grad_head, &vh.transpose());
            let mut grad_scores = softmax_backward(&attn, &grad_attn);
            for s in grad_scores.data.iter_mut() {
                *s *= scale;
            }
            let grad_qh = Matrix::matmul_views(&grad_scores, &kh);
            let grad_kh = Matrix::matmul_views(&grad_scores.transpose(), &qh);
            copy_into(&mut grad_q, h * head_dim, &grad_qh);
            copy_into(&mut grad_k, h * head_dim, &grad_kh);
        }
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
        self.wq.adam_step(lr, beta1, beta2, eps, weight_decay);
        self.wk.adam_step(lr, beta1, beta2, eps, weight_decay);
        self.wv.adam_step(lr, beta1, beta2, eps, weight_decay);
        self.wo.adam_step(lr, beta1, beta2, eps, weight_decay);
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

    fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        MultiHeadAttentionT::adam_step(self, lr, beta1, beta2, eps, weight_decay);
    }

    fn parameters(&mut self) -> Vec<&mut LinearT> {
        MultiHeadAttentionT::parameters(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_masking_keeps_tokens_separate() {
        let mut attn = MultiHeadAttentionT::new(4, 2);
        // set all weights to identity so attention operates on input directly
        for i in 0..4 {
            for j in 0..4 {
                let v = if i == j { 1.0 } else { 0.0 };
                let idx = i * 4 + j;
                attn.wq.w.data[idx] = v;
                attn.wk.w.data[idx] = v;
                attn.wv.w.data[idx] = v;
                attn.wo.w.data[idx] = v;
            }
        }
        let x = Matrix::from_vec(2, 4, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        // mask off-diagonal to prevent interaction between tokens
        let mask = Matrix::from_vec(2, 2, vec![0.0, -1e9, -1e9, 0.0]);
        attn.set_mask(mask);
        let out = attn.forward(&Tensor::from_matrix(x.clone()));
        assert_eq!(out.shape[0], 2);
        assert_eq!(out.shape[1], 4);
        for i in 0..out.data.len() {
            assert!((out.data[i] - x.data[i]).abs() < 1e-6);
        }
    }
}
