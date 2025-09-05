use crate::layers::sigmoid;
use crate::layers::{EmbeddingT, LinearT, MultiHeadAttentionT};
use crate::math::Matrix;
use crate::tensor::Tensor;

/// Configuration for building a [`LlamaModel`].
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
}

/// Root mean square layer normalization.
pub struct RMSNorm {
    pub weight: Vec<f32>,
    grad: Vec<f32>,
    eps: f32,
    x: Matrix,
    rms: Vec<f32>,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            weight: vec![1.0; dim],
            grad: vec![0.0; dim],
            eps,
            x: Matrix::zeros(0, 0),
            rms: Vec::new(),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let rows = x.shape[0];
        let cols = x.shape[1];
        let mut out = Matrix::zeros(rows, cols);
        for r in 0..rows {
            let mut sum = 0.0;
            for c in 0..cols {
                let v = x.data[r * cols + c];
                sum += v * v;
            }
            let rms = (sum / cols as f32 + self.eps).sqrt();
            for c in 0..cols {
                let v = x.data[r * cols + c] / rms * self.weight[c];
                out.data[r * cols + c] = v;
            }
        }
        Tensor::from_matrix(out)
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        let rows = x.rows;
        let cols = x.cols;
        self.x = x.clone();
        self.rms = Vec::with_capacity(rows);
        let mut out = Matrix::zeros(rows, cols);
        for r in 0..rows {
            let mut sum = 0.0;
            for c in 0..cols {
                let v = x.data[r * cols + c];
                sum += v * v;
            }
            let rms = (sum / cols as f32 + self.eps).sqrt();
            self.rms.push(rms);
            for c in 0..cols {
                let v = x.data[r * cols + c] / rms * self.weight[c];
                out.data[r * cols + c] = v;
            }
        }
        out
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let rows = grad_out.rows;
        let cols = grad_out.cols;
        let mut grad_in = Matrix::zeros(rows, cols);
        for r in 0..rows {
            let rms = self.rms[r];
            let mut dot = 0.0;
            for c in 0..cols {
                let go = grad_out.data[r * cols + c];
                let x = self.x.data[r * cols + c];
                self.grad[c] += go * x / rms;
                dot += go * self.weight[c] * x;
            }
            let coeff = dot / (rms * rms * rms * cols as f32);
            for c in 0..cols {
                let go = grad_out.data[r * cols + c];
                let x = self.x.data[r * cols + c];
                let g = go * self.weight[c] / rms - x * coeff;
                grad_in.data[r * cols + c] = g;
            }
        }
        grad_in
    }

    pub fn zero_grad(&mut self) {
        for g in self.grad.iter_mut() {
            *g = 0.0;
        }
    }

    pub fn adam_step(&mut self, _lr: f32, _b1: f32, _b2: f32, _eps: f32, _wd: f32) {
        // optimiser placeholder
    }
}

/// Simple gated feed-forward network used in LLaMA blocks.
pub struct GatedFFN {
    pub w1: LinearT,
    pub w2: LinearT,
    pub w3: LinearT,
    gate: Matrix,
    up: Matrix,
    act: Matrix,
}

impl GatedFFN {
    pub fn new(dim: usize, hidden: usize) -> Self {
        Self {
            w1: LinearT::new(dim, hidden),
            w2: LinearT::new(hidden, dim),
            w3: LinearT::new(dim, hidden),
            gate: Matrix::zeros(0, 0),
            up: Matrix::zeros(0, 0),
            act: Matrix::zeros(0, 0),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut gate = self.w1.forward(x);
        sigmoid::forward_tensor(&mut gate);
        let up = self.w3.forward(x);
        let mut prod = gate.clone();
        for (p, u) in prod.data.iter_mut().zip(up.data.iter()) {
            *p *= *u;
        }
        self.w2.forward(&prod)
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        self.gate = self.w1.forward_local(x);
        self.act = self.gate.clone();
        sigmoid::forward_matrix(&mut self.act);
        self.up = self.w3.forward_local(x);
        let mut prod = Matrix::zeros(self.act.rows, self.act.cols);
        for i in 0..prod.data.len() {
            prod.data[i] = self.act.data[i] * self.up.data[i];
        }
        self.w2.forward_local(&prod)
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let grad_prod = self.w2.backward(grad_out);
        let mut grad_act = Matrix::zeros(self.act.rows, self.act.cols);
        let mut grad_up = Matrix::zeros(self.up.rows, self.up.cols);
        for i in 0..grad_prod.data.len() {
            grad_act.data[i] = grad_prod.data[i] * self.up.data[i];
            grad_up.data[i] = grad_prod.data[i] * self.act.data[i];
        }
        let mut grad_gate = grad_act.clone();
        sigmoid::backward(&mut grad_gate, &self.act);
        let gx1 = self.w1.backward(&grad_gate);
        let gx3 = self.w3.backward(&grad_up);
        Matrix::add(&gx1, &gx3)
    }

    pub fn zero_grad(&mut self) {
        self.w1.zero_grad();
        self.w2.zero_grad();
        self.w3.zero_grad();
    }

    pub fn adam_step(&mut self, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32) {
        self.w1.adam_step(lr, b1, b2, eps, wd);
        self.w2.adam_step(lr, b1, b2, eps, wd);
        self.w3.adam_step(lr, b1, b2, eps, wd);
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let (w1, w2, w3) = (&mut self.w1, &mut self.w2, &mut self.w3);
        vec![w1, w2, w3]
    }
}

fn apply_rotary_matrix(x: &mut Matrix) {
    let seq_len = x.rows;
    let dim = x.cols;
    let half = dim / 2;
    for p in 0..seq_len {
        for i in 0..half {
            let angle = (p as f32) / 10000f32.powf((2 * i) as f32 / dim as f32);
            let (sin, cos) = angle.sin_cos();
            let idx1 = p * dim + i;
            let idx2 = p * dim + i + half;
            let v1 = x.data[idx1];
            let v2 = x.data[idx2];
            x.data[idx1] = v1 * cos - v2 * sin;
            x.data[idx2] = v1 * sin + v2 * cos;
        }
    }
}

fn apply_rotary_tensor(t: &Tensor) -> Tensor {
    let mut m = Matrix::from_vec(t.shape[0], t.shape[1], t.data.clone());
    apply_rotary_matrix(&mut m);
    Tensor::from_matrix(m)
}

/// A single LLaMA transformer block.
pub struct LlamaBlock {
    pub attn: MultiHeadAttentionT,
    pub ffn: GatedFFN,
    pub norm1: RMSNorm,
    pub norm2: RMSNorm,
    x: Matrix,
    attn_out: Matrix,
    res1: Matrix,
    norm2_out: Matrix,
    ff_out: Matrix,
    res2: Matrix,
}

impl LlamaBlock {
    pub fn new(model_dim: usize, num_heads: usize, ff_hidden: usize) -> Self {
        Self {
            attn: MultiHeadAttentionT::new(model_dim, num_heads),
            ffn: GatedFFN::new(model_dim, ff_hidden),
            norm1: RMSNorm::new(model_dim, 1e-6),
            norm2: RMSNorm::new(model_dim, 1e-6),
            x: Matrix::zeros(0, 0),
            attn_out: Matrix::zeros(0, 0),
            res1: Matrix::zeros(0, 0),
            norm2_out: Matrix::zeros(0, 0),
            ff_out: Matrix::zeros(0, 0),
            res2: Matrix::zeros(0, 0),
        }
    }

    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        let h1 = self.norm1.forward(x);
        let h1 = apply_rotary_tensor(&h1);
        let attn = self.attn.forward(&h1);
        let res1 = Tensor::add(&attn, x);
        let n2 = self.norm2.forward(&res1);
        let ff = self.ffn.forward(&n2);
        Tensor::add(&ff, &res1)
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        self.x = x.clone();
        let mut h1 = self.norm1.forward_train(x);
        apply_rotary_matrix(&mut h1);
        self.attn_out = self.attn.forward_train(&h1);
        self.res1 = Matrix::add(&self.attn_out, x);
        self.norm2_out = self.norm2.forward_train(&self.res1);
        self.ff_out = self.ffn.forward_train(&self.norm2_out);
        self.res2 = Matrix::add(&self.ff_out, &self.res1);
        self.res2.clone()
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let mut g = self.ffn.backward(grad_out);
        g = Matrix::add(&g, grad_out);
        g = self.norm2.backward(&g);
        let g_attn = self.attn.backward(&g);
        let mut g_x = Matrix::add(&g, &g_attn);
        self.norm1.backward(&g_x)
    }

    pub fn zero_grad(&mut self) {
        self.attn.zero_grad();
        self.ffn.zero_grad();
        self.norm1.zero_grad();
        self.norm2.zero_grad();
    }

    pub fn adam_step(&mut self, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32) {
        self.attn.adam_step(lr, b1, b2, eps, wd);
        self.ffn.adam_step(lr, b1, b2, eps, wd);
        self.norm1.adam_step(lr, b1, b2, eps, wd);
        self.norm2.adam_step(lr, b1, b2, eps, wd);
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let mut params = self.attn.parameters();
        params.extend(self.ffn.parameters());
        params
    }
}

/// Simplified LLaMA model composed of an embedding layer and a stack of blocks.
pub struct LlamaModel {
    pub layers: Vec<LlamaBlock>,
    pub embedding: EmbeddingT,
}

impl LlamaModel {
    pub fn new(cfg: LlamaConfig) -> Self {
        let mut layers = Vec::new();
        for _ in 0..cfg.num_layers {
            layers.push(LlamaBlock::new(
                cfg.hidden_size,
                cfg.num_heads,
                cfg.intermediate_size,
            ));
        }
        Self {
            layers,
            embedding: EmbeddingT::new(cfg.vocab_size, cfg.hidden_size),
        }
    }

    pub fn forward(&mut self, one_hot_x: Matrix) -> Tensor {
        let mut h = Tensor::from_matrix(one_hot_x);
        h = self.embedding.forward(&h);
        for layer in self.layers.iter_mut() {
            h = layer.forward(&h);
        }
        h
    }

    pub fn forward_train(&mut self, one_hot_x: &Matrix) -> Matrix {
        let mut h = self.embedding.forward_train(one_hot_x);
        for layer in self.layers.iter_mut() {
            h = layer.forward_train(&h);
        }
        h
    }

    pub fn backward(&mut self, grad_out: &Matrix) {
        let mut g = grad_out.clone();
        for layer in self.layers.iter_mut().rev() {
            g = layer.backward(&g);
        }
        self.embedding.backward(&g);
    }

    pub fn zero_grad(&mut self) {
        self.embedding.zero_grad();
        for l in self.layers.iter_mut() {
            l.zero_grad();
        }
    }

    pub fn adam_step(&mut self, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32) {
        self.embedding.adam_step(lr, b1, b2, eps, wd);
        for l in self.layers.iter_mut() {
            l.adam_step(lr, b1, b2, eps, wd);
        }
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let mut params = self.embedding.parameters();
        for l in self.layers.iter_mut() {
            params.extend(l.parameters());
        }
        params
    }
}
