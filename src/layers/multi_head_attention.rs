use crate::tensor::Tensor;
use crate::math::Matrix;
use super::linear::LinearT;
use super::layer::Layer;

fn slice_cols(m: &Matrix, start: usize, width: usize) -> Matrix {
    let mut out = Matrix::zeros(m.rows, width);
    for r in 0..m.rows {
        let src = &m.data[r * m.cols + start..r * m.cols + start + width];
        let dst = &mut out.data[r * width..(r + 1) * width];
        dst.copy_from_slice(src);
    }
    out
}

fn slice_rows(m: &Matrix, start: usize, count: usize) -> Matrix {
    let mut out = Matrix::zeros(count, m.cols);
    for r in 0..count {
        let src = &m.data[(start + r) * m.cols..(start + r + 1) * m.cols];
        let dst = &mut out.data[r * m.cols..(r + 1) * m.cols];
        dst.copy_from_slice(src);
    }
    out
}

fn copy_into(dest: &mut Matrix, start_col: usize, src: &Matrix) {
    for r in 0..dest.rows {
        let dst = &mut dest.data[r * dest.cols + start_col..r * dest.cols + start_col + src.cols];
        let s = &src.data[r * src.cols..(r + 1) * src.cols];
        dst.copy_from_slice(s);
    }
}

fn softmax_backward(attn: &Matrix, grad_out: &Matrix) -> Matrix {
    let mut grad = Matrix::zeros(grad_out.rows, grad_out.cols);
    for r in 0..grad_out.rows {
        let row_start = r * grad_out.cols;
        let g_row = &grad_out.data[row_start..row_start + grad_out.cols];
        let a_row = &attn.data[row_start..row_start + grad_out.cols];
        let mut dot = 0.0;
        for c in 0..grad_out.cols {
            dot += g_row[c] * a_row[c];
        }
        for c in 0..grad_out.cols {
            grad.data[row_start + c] = a_row[c] * (g_row[c] - dot);
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
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);
        let seq_len = q.data.rows;
        let model_dim = q.data.cols;
        let head_dim = model_dim / self.num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut concat = Matrix::zeros(seq_len, model_dim);
        for h in 0..self.num_heads {
            let qh = slice_cols(&q.data, h * head_dim, head_dim);
            let kh = slice_cols(&k.data, h * head_dim, head_dim);
            let vh = slice_cols(&v.data, h * head_dim, head_dim);
            let mut scores = Matrix::matmul(&qh, &kh.transpose());
            for s in scores.data.iter_mut() {
                *s *= scale;
            }
            if let Some(m) = self.mask.as_ref() {
                for i in 0..scores.data.len() {
                    scores.data[i] += m.data[i];
                }
            }
            let attn = scores.softmax();
            let head_out = Matrix::matmul(&attn, &vh);
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
            let qh = slice_cols(&self.q, h * head_dim, head_dim);
            let kh = slice_cols(&self.k, h * head_dim, head_dim);
            let vh = slice_cols(&self.v, h * head_dim, head_dim);
            let mut scores = Matrix::matmul(&qh, &kh.transpose());
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
                let dst = &mut self.attn.data[(h * seq_len + r) * seq_len..(h * seq_len + r + 1) * seq_len];
                let src = &attn.data[r * seq_len..(r + 1) * seq_len];
                dst.copy_from_slice(src);
            }
            let head_out = Matrix::matmul(&attn, &vh);
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
            let qh = slice_cols(&self.q, h * head_dim, head_dim);
            let kh = slice_cols(&self.k, h * head_dim, head_dim);
            let vh = slice_cols(&self.v, h * head_dim, head_dim);
            let attn = slice_rows(&self.attn, h * seq_len, seq_len);
            let grad_head = slice_cols(&grad_concat, h * head_dim, head_dim);
            let grad_vh = Matrix::matmul(&attn.transpose(), &grad_head);
            copy_into(&mut grad_v, h * head_dim, &grad_vh);
            let grad_attn = Matrix::matmul(&grad_head, &vh.transpose());
            let mut grad_scores = softmax_backward(&attn, &grad_attn);
            for s in grad_scores.data.iter_mut() {
                *s *= scale;
            }
            let grad_qh = Matrix::matmul(&grad_scores, &kh);
            let grad_kh = Matrix::matmul(&grad_scores.transpose(), &qh);
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
            let qh = slice_cols(&self.q, h * head_dim, head_dim);
            let kh = slice_cols(&self.k, h * head_dim, head_dim);
            let vh = slice_cols(&self.v, h * head_dim, head_dim);
            let attn = slice_rows(&self.attn, h * seq_len, seq_len);
            let grad_head = slice_cols(&grad_concat, h * head_dim, head_dim);
            let grad_vh = Matrix::matmul(&attn.transpose(), &grad_head);
            copy_into(&mut grad_v, h * head_dim, &grad_vh);
            let grad_attn = Matrix::matmul(&grad_head, &vh.transpose());
            let mut grad_scores = softmax_backward(&attn, &grad_attn);
            for s in grad_scores.data.iter_mut() {
                *s *= scale;
            }
            let grad_qh = Matrix::matmul(&grad_scores, &kh);
            let grad_kh = Matrix::matmul(&grad_scores.transpose(), &qh);
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
                attn.wq.w.data.set(i, j, v);
                attn.wk.w.data.set(i, j, v);
                attn.wv.w.data.set(i, j, v);
                attn.wo.w.data.set(i, j, v);
            }
        }
        let x = Matrix::from_vec(2, 4, vec![1.0, 0.0, 0.0, 0.0,
                                            0.0, 1.0, 0.0, 0.0]);
        // mask off-diagonal to prevent interaction between tokens
        let mask = Matrix::from_vec(2, 2, vec![0.0, -1e9,
                                              -1e9, 0.0]);
        attn.set_mask(mask);
        let out = attn.forward(&Tensor::from_matrix(x.clone()));
        assert_eq!(out.data.rows, 2);
        assert_eq!(out.data.cols, 4);
        for i in 0..out.data.data.len() {
            assert!((out.data.data[i] - x.data[i]).abs() < 1e-6);
        }
    }
}

