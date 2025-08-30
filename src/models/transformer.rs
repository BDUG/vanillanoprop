use crate::layers::{Activation, Dropout, EmbeddingT, FeedForwardT, LayerNorm, LinearT, MultiHeadAttentionT};
use crate::math::Matrix;
use crate::positional::positional_encoding;
use crate::tensor::Tensor;

/// Single Transformer encoder layer with pre-layer normalization, dropout and residual
/// connections.
pub struct TransformerEncoderLayer {
    pub attn: MultiHeadAttentionT,
    pub ff: FeedForwardT,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    dropout1: Dropout,
    dropout2: Dropout,
    p_drop: f32,
    // caches for backward
    x: Matrix,
    attn_out: Matrix,
    res1: Matrix,
    norm1_out: Matrix,
    ff_out: Matrix,
    res2: Matrix,
}

impl TransformerEncoderLayer {
    /// Create a new encoder layer.
    pub fn new(model_dim: usize, num_heads: usize, ff_hidden: usize, activation: Activation, p_drop: f32) -> Self {
        Self {
            attn: MultiHeadAttentionT::new(model_dim, num_heads),
            ff: FeedForwardT::new(model_dim, ff_hidden, activation),
            norm1: LayerNorm::new(model_dim, 1e-5),
            norm2: LayerNorm::new(model_dim, 1e-5),
            dropout1: Dropout::new(),
            dropout2: Dropout::new(),
            p_drop,
            x: Matrix::zeros(0, 0),
            attn_out: Matrix::zeros(0, 0),
            res1: Matrix::zeros(0, 0),
            norm1_out: Matrix::zeros(0, 0),
            ff_out: Matrix::zeros(0, 0),
            res2: Matrix::zeros(0, 0),
        }
    }

    /// Inference path. Applies optional attention mask.
    pub fn forward(&mut self, x: &Tensor, mask: Option<&Matrix>) -> Tensor {
        if let Some(m) = mask {
            self.attn.set_mask(m.clone());
        } else {
            self.attn.clear_mask();
        }
        let attn_out = self.attn.forward(x);
        let res1 = Tensor::add(&attn_out, x);
        let norm1 = self.norm1.forward(&res1);
        let ff = self.ff.forward(&norm1);
        let res2 = Tensor::add(&ff, &norm1);
        self.norm2.forward(&res2)
    }

    /// Training forward pass with dropout and residual connections.
    pub fn forward_train(&mut self, x: &Matrix, mask: Option<&Matrix>) -> Matrix {
        self.x = x.clone();
        if let Some(m) = mask {
            self.attn.set_mask(m.clone());
        } else {
            self.attn.clear_mask();
        }
        self.attn_out = self.attn.forward_train(x);
        self.attn_out = self.dropout1.forward(&self.attn_out, self.p_drop, true);
        self.res1 = Matrix::add(&self.attn_out, x);
        self.norm1_out = self.norm1.forward_train(&self.res1);
        self.ff_out = self.ff.forward_train(&self.norm1_out);
        self.ff_out = self.dropout2.forward(&self.ff_out, self.p_drop, true);
        self.res2 = Matrix::add(&self.ff_out, &self.norm1_out);
        self.norm2.forward_train(&self.res2)
    }

    /// Backward pass returning gradient with respect to the input.
    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let g_norm2 = self.norm2.backward(grad_out);
        let g_res2 = g_norm2;
        let g_ff_part = g_res2.clone();
        let g_h1_part = g_res2.clone();
        let g_ff_drop = self.dropout2.backward(&g_ff_part);
        let g_ff = self.ff.backward(&g_ff_drop);
        let mut g_norm1 = Matrix::add(&g_ff, &g_h1_part);
        g_norm1 = self.norm1.backward(&g_norm1);
        let g_res1 = g_norm1;
        let g_attn_part = g_res1.clone();
        let mut g_x = g_res1.clone();
        let g_attn_drop = self.dropout1.backward(&g_attn_part);
        let g_attn = self.attn.backward(&g_attn_drop);
        g_x = Matrix::add(&g_x, &g_attn);
        g_x
    }

    pub fn zero_grad(&mut self) {
        self.attn.zero_grad();
        self.ff.zero_grad();
        self.norm1.zero_grad();
        self.norm2.zero_grad();
    }

    pub fn adam_step(&mut self, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32) {
        self.attn.adam_step(lr, b1, b2, eps, wd);
        self.ff.adam_step(lr, b1, b2, eps, wd);
        self.norm1.adam_step(lr, b1, b2, eps, wd);
        self.norm2.adam_step(lr, b1, b2, eps, wd);
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let mut params = self.attn.parameters();
        params.extend(self.ff.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
    }
}

/// Full Transformer encoder with token embedding and positional encoding.
pub struct TransformerEncoder {
    pub layers: Vec<TransformerEncoderLayer>,
    pub embedding: EmbeddingT,
    pos: Matrix,
}

impl TransformerEncoder {
    /// Build a Transformer encoder.
    pub fn new(n_layers: usize, vocab_size: usize, model_dim: usize, num_heads: usize, ff_hidden: usize, dropout: f32) -> Self {
        let mut layers = Vec::new();
        for _ in 0..n_layers { 
            layers.push(TransformerEncoderLayer::new(model_dim, num_heads, ff_hidden, Activation::ReLU, dropout));
        }
        Self { layers, embedding: EmbeddingT::new(vocab_size, model_dim), pos: Matrix::zeros(0,0) }
    }

    /// Inference forward pass.
    pub fn forward(&mut self, one_hot_x: Matrix, mask: Option<&Matrix>) -> Tensor {
        let mut h = Tensor::from_matrix(one_hot_x);
        h = self.embedding.forward(&h);
        let pos = positional_encoding(h.shape[0], h.shape[1]);
        let p = Tensor::from_matrix(pos);
        h = Tensor::add(&h, &p);
        for layer in self.layers.iter_mut() {
            h = layer.forward(&h, mask);
        }
        h
    }

    /// Training forward pass.
    pub fn forward_train(&mut self, one_hot_x: &Matrix, mask: Option<&Matrix>) -> Matrix {
        let mut h = self.embedding.forward_train(one_hot_x);
        self.pos = positional_encoding(h.rows, h.cols);
        h = Matrix::add(&h, &self.pos);
        for layer in self.layers.iter_mut() {
            h = layer.forward_train(&h, mask);
        }
        h
    }

    /// Backpropagate through the encoder.
    pub fn backward(&mut self, grad_out: &Matrix) {
        let mut g = grad_out.clone();
        for layer in self.layers.iter_mut().rev() {
            g = layer.backward(&g);
        }
        self.embedding.backward(&g);
    }

    pub fn zero_grad(&mut self) {
        self.embedding.zero_grad();
        for l in self.layers.iter_mut() { l.zero_grad(); }
    }

    pub fn adam_step(&mut self, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32) {
        self.embedding.adam_step(lr, b1, b2, eps, wd);
        for l in self.layers.iter_mut() { l.adam_step(lr, b1, b2, eps, wd); }
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let mut params = self.embedding.parameters();
        for l in self.layers.iter_mut() { params.extend(l.parameters()); }
        params
    }
}

// Optional decoder can be implemented later if needed.
