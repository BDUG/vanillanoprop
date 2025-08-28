use crate::layers::{
    Activation, EmbeddingT, FeedForwardT, Layer, LinearT, MixtureOfExpertsT, MultiHeadAttentionT,
};
use crate::math::Matrix;
use crate::model::Model;
use crate::positional::positional_encoding;
use crate::tensor::Tensor;

pub struct EncoderLayerT {
    pub attn: Box<dyn Layer>,
    pub ff: Box<dyn Layer>,
    attn_out: Matrix,
}

impl EncoderLayerT {
    pub fn new(
        dim: usize,
        hidden: usize,
        activation: Activation,
        moe: bool,
        num_experts: usize,
    ) -> Self {
        let ff: Box<dyn Layer> = if moe {
            let mut experts: Vec<Box<dyn Layer>> = Vec::new();
            for _ in 0..num_experts.max(1) {
                experts.push(Box::new(FeedForwardT::new(dim, hidden, activation)));
            }
            Box::new(MixtureOfExpertsT::new(dim, experts, num_experts))
        } else {
            Box::new(FeedForwardT::new(dim, hidden, activation))
        };
        Self {
            attn: Box::new(MultiHeadAttentionT::new(dim, 1)),
            ff,
            attn_out: Matrix::zeros(0, 0),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.attn.forward(x);
        self.ff.forward(&h)
    }

    pub fn forward_local(&mut self, x: &Matrix) -> Matrix {
        self.attn_out = self.attn.forward_train(x);
        self.ff.forward_train(&self.attn_out)
    }

    pub fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        let grad_attn_out = self.ff.fa_update(grad_out, lr);
        self.attn.fa_update(&grad_attn_out, lr)
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        self.forward_local(x)
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let grad_attn_out = self.ff.backward(grad_out);
        self.attn.backward(&grad_attn_out)
    }

    pub fn zero_grad(&mut self) {
        self.attn.zero_grad();
        self.ff.zero_grad();
    }

    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.attn.adam_step(lr, beta1, beta2, eps, weight_decay);
        self.ff.adam_step(lr, beta1, beta2, eps, weight_decay);
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let mut params = self.attn.parameters();
        params.extend(self.ff.parameters());
        params
    }
}

pub struct EncoderT {
    pub layers: Vec<EncoderLayerT>,
    pub embedding: Box<dyn Layer>,
    pos: Matrix,
}

impl EncoderT {
    pub fn new(
        n: usize,
        vocab_size: usize,
        model_dim: usize,
        hidden: usize,
        activation: Activation,
        moe: bool,
        num_experts: usize,
    ) -> Self {
        let mut v = Vec::new();
        for _ in 0..n {
            v.push(EncoderLayerT::new(
                model_dim,
                hidden,
                activation,
                moe,
                num_experts,
            ));
        }
        Self {
            layers: v,
            embedding: Box::new(EmbeddingT::new(vocab_size, model_dim)),
            pos: Matrix::zeros(0, 0),
        }
    }

    pub fn forward(&self, x: Matrix) -> Tensor {
        // existing inference path
        let mut h = Tensor::from_matrix(x);
        h = self.embedding.forward(&h);
        let pos = positional_encoding(h.data.rows, h.data.cols);
        let p = Tensor::from_matrix(pos);
        h = Tensor::add(&h, &p);
        for l in &self.layers {
            h = l.forward(&h);
        }
        h
    }

    pub fn forward_local(&mut self, x: &Matrix) -> Matrix {
        let mut h = self.embedding.forward_train(x);
        self.pos = positional_encoding(h.rows, h.cols);
        h = Matrix::add(&h, &self.pos);
        for l in self.layers.iter_mut() {
            h = l.forward_local(&h);
        }
        h
    }

    pub fn fa_update(&mut self, grad_out: &Matrix, lr: f32) {
        let mut g = grad_out.clone();
        for l in self.layers.iter_mut().rev() {
            g = l.fa_update(&g, lr);
        }
        self.embedding.fa_update(&g, lr);
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        self.forward_local(x)
    }

    pub fn backward(&mut self, grad_out: &Matrix) {
        let mut g = grad_out.clone();
        for l in self.layers.iter_mut().rev() {
            g = l.backward(&g);
        }
        self.embedding.backward(&g);
    }

    pub fn zero_grad(&mut self) {
        self.embedding.zero_grad();
        for l in self.layers.iter_mut() {
            l.zero_grad();
        }
    }

    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.embedding
            .adam_step(lr, beta1, beta2, eps, weight_decay);
        for l in self.layers.iter_mut() {
            l.adam_step(lr, beta1, beta2, eps, weight_decay);
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

/// Build an encoder architecture as a [`Model`] graph. The returned
/// model contains an input embedding followed by `n` attention +
/// feed-forward blocks connected sequentially.
pub fn encoder_model(n: usize) -> Model {
    let mut m = Model::new();
    let input = m.add("input");
    let embedding = m.add("embedding");
    m.connect(input, embedding);
    let mut prev = embedding;
    for i in 0..n {
        let attn = m.add(format!("attn{}", i));
        let ff = m.add(format!("ff{}", i));
        m.connect(prev, attn);
        m.connect(attn, ff);
        prev = ff;
    }
    m
}
