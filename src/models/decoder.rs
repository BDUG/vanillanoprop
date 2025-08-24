use crate::layers::{EmbeddingT, FeedForwardT, LinearT, MultiHeadAttentionT, Layer};
use crate::math::Matrix;
use crate::autograd::Tensor;

pub struct DecoderLayerT {
    self_attn: Box<dyn Layer>,
    enc_dec_attn: Box<dyn Layer>,
    ff: Box<dyn Layer>,
    h1: Matrix,
    ctx: Matrix,
}

impl DecoderLayerT {
    pub fn new(dim: usize, hidden: usize) -> Self {
        Self {
            self_attn: Box::new(MultiHeadAttentionT::new(dim)),
            enc_dec_attn: Box::new(MultiHeadAttentionT::new(dim)),
            ff: Box::new(FeedForwardT::new(dim, hidden)),
            h1: Matrix::zeros(0, 0),
            ctx: Matrix::zeros(0, 0),
        }
    }

    pub fn forward(&self, x: &Tensor, enc_out: &Tensor) -> Tensor {
        let h1 = self.self_attn.forward(x);
        let ctx = if h1.data.rows == enc_out.data.rows && h1.data.cols == enc_out.data.cols {
            Tensor::add(&h1, enc_out)
        } else {
            h1.clone()
        };
        let h2 = self.enc_dec_attn.forward(&ctx);
        self.ff.forward(&h2)
    }

    pub fn forward_train(&mut self, x: &Matrix, enc_out: &Matrix) -> Matrix {
        self.h1 = self.self_attn.forward_train(x);
        if self.h1.rows == enc_out.rows && self.h1.cols == enc_out.cols {
            self.ctx = Matrix::add(&self.h1, enc_out);
        } else {
            self.ctx = self.h1.clone();
        }
        let h2 = self.enc_dec_attn.forward_train(&self.ctx);
        self.ff.forward_train(&h2)
    }

    /// Backward pass returns gradient w.r.t. the layer input and w.r.t. the
    /// encoder context that was added to the decoder state.
    pub fn backward(&mut self, grad_out: &Matrix) -> (Matrix, Matrix) {
        let grad_ff = self.ff.backward(grad_out);
        let grad_ctx = self.enc_dec_attn.backward(&grad_ff);
        let grad_h1 = grad_ctx.clone();
        let grad_enc = if self.h1.rows == self.ctx.rows && self.h1.cols == self.ctx.cols {
            grad_ctx.clone()
        } else {
            Matrix::zeros(grad_ctx.rows, grad_ctx.cols)
        };
        let grad_in = self.self_attn.backward(&grad_h1);
        (grad_in, grad_enc)
    }

    pub fn zero_grad(&mut self) {
        self.self_attn.zero_grad();
        self.enc_dec_attn.zero_grad();
        self.ff.zero_grad();
    }

    pub fn adam_step(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) {
        self.self_attn
            .adam_step(lr, beta1, beta2, eps, weight_decay);
        self.enc_dec_attn
            .adam_step(lr, beta1, beta2, eps, weight_decay);
        self.ff
            .adam_step(lr, beta1, beta2, eps, weight_decay);
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let mut params = self.self_attn.parameters();
        params.extend(self.enc_dec_attn.parameters());
        params.extend(self.ff.parameters());
        params
    }
}

pub struct DecoderT {
    pub layers: Vec<DecoderLayerT>,
    pub embedding: Box<dyn Layer>,
    pub proj: Box<dyn Layer>,
    enc_out_cache: Matrix,
}

impl DecoderT {
    pub fn new(n: usize, vocab_size: usize, model_dim: usize, hidden: usize) -> Self {
        let mut v = Vec::new();
        for _ in 0..n {
            v.push(DecoderLayerT::new(model_dim, hidden));
        }
        Self {
            layers: v,
            embedding: Box::new(EmbeddingT::new(vocab_size, model_dim)),
            proj: Box::new(LinearT::new(model_dim, vocab_size)),
            enc_out_cache: Matrix::zeros(0, 0),
        }
    }

    pub fn forward(&self, one_hot_x: &Tensor, enc_out: &Tensor) -> Tensor {
        // one_hot_x = (seq_len × vocab_size)
        let mut h = self.embedding.forward(one_hot_x);
        for l in &self.layers {
            h = l.forward(&h, enc_out);
        }
        self.proj.forward(&h)
    }

    pub fn forward_train(&mut self, one_hot_x: &Matrix, enc_out: &Matrix) -> Matrix {
        let mut h = self.embedding.forward_train(one_hot_x);
        for l in self.layers.iter_mut() {
            h = l.forward_train(&h, enc_out);
        }
        self.enc_out_cache = enc_out.clone();
        self.proj.forward_train(&h)
    }

    /// Backward pass through the decoder.  Returns the gradient with respect to
    /// the encoder output so that it can be backpropagated through the encoder
    /// as well.
    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let mut g = self.proj.backward(grad_out);
        let mut grad_enc = Matrix::zeros(self.enc_out_cache.rows, self.enc_out_cache.cols);
        for l in self.layers.iter_mut().rev() {
            let (ng, genc) = l.backward(&g);
            g = ng;
            grad_enc = Matrix::add(&grad_enc, &genc);
        }
        self.embedding.backward(&g);
        grad_enc
    }

    pub fn zero_grad(&mut self) {
        self.embedding.zero_grad();
        self.proj.zero_grad();
        for l in self.layers.iter_mut() {
            l.zero_grad();
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
        self.embedding
            .adam_step(lr, beta1, beta2, eps, weight_decay);
        self.proj
            .adam_step(lr, beta1, beta2, eps, weight_decay);
        for l in self.layers.iter_mut() {
            l.adam_step(lr, beta1, beta2, eps, weight_decay);
        }
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let mut params = self.embedding.parameters();
        params.extend(self.proj.parameters());
        for l in self.layers.iter_mut() {
            params.extend(l.parameters());
        }
        params
    }
}

