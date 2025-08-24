use crate::autograd::Tensor;
use crate::math::Matrix;
use crate::positional::positional_encoding;

pub struct LinearT {
    pub w: Tensor,
}

impl LinearT {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let data = Matrix::from_vec(
            in_dim,
            out_dim,
            (0..in_dim * out_dim)
                .map(|_| (rand::random::<f32>() - 0.5) * 0.02)
                .collect(),
        );
        Self {
            w: Tensor::from_matrix(data),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::matmul(x, &self.w)
    }
}

/// Embedding layer: maps one-hot (vocab_size) into dense model_dim.
pub struct EmbeddingT {
    pub table: LinearT, // weight matrix (vocab_size x model_dim)
}

impl EmbeddingT {
    pub fn new(vocab_size: usize, model_dim: usize) -> Self {
        Self {
            table: LinearT::new(vocab_size, model_dim),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x: (seq_len x vocab_size) → multiply with (vocab_size x model_dim)
        Tensor::matmul(x, &self.table.w)
    }
}

pub struct FeedForwardT {
    pub w1: LinearT,
    pub w2: LinearT,
}

impl FeedForwardT {
    pub fn new(dim: usize, hidden: usize) -> Self {
        Self {
            w1: LinearT::new(dim, hidden),
            w2: LinearT::new(hidden, dim),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = self.w1.forward(x);
        for v in h.data.data.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        self.w2.forward(&h)
    }
}

pub struct MultiHeadAttentionT {
    pub wq: LinearT,
    pub wk: LinearT,
    pub wv: LinearT,
    pub wo: LinearT,
}

impl MultiHeadAttentionT {
    pub fn new(model_dim: usize) -> Self {
        Self {
            wq: LinearT::new(model_dim, model_dim),
            wk: LinearT::new(model_dim, model_dim),
            wv: LinearT::new(model_dim, model_dim),
            wo: LinearT::new(model_dim, model_dim),
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
}

pub struct EncoderLayerT {
    pub attn: MultiHeadAttentionT,
    pub ff: FeedForwardT,
}

impl EncoderLayerT {
    pub fn new(dim: usize, hidden: usize) -> Self {
        Self {
            attn: MultiHeadAttentionT::new(dim),
            ff: FeedForwardT::new(dim, hidden),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.attn.forward(x);
        self.ff.forward(&h)
    }
}

pub struct EncoderT {
    pub layers: Vec<EncoderLayerT>,
    pub embedding: EmbeddingT,
}

impl EncoderT {
    pub fn new(n: usize, vocab_size: usize, model_dim: usize, hidden: usize) -> Self {
        let mut v = Vec::new();
        for _ in 0..n {
            v.push(EncoderLayerT::new(model_dim, hidden));
        }
        Self {
            layers: v,
            embedding: EmbeddingT::new(vocab_size, model_dim),
        }
    }

    pub fn forward(&self, x: &Matrix) -> Tensor {
        // x = one-hot (seq_len x vocab_size)
        let mut h = Tensor::from_matrix(x.clone());
        h = self.embedding.forward(&h); // now (seq_len x model_dim)

        // positional encoding & transformer layers
        let pos = positional_encoding(h.data.rows, h.data.cols);
        let p = Tensor::from_matrix(pos);
        h = Tensor::add(&h, &p);
        for l in &self.layers {
            h = l.forward(&h);
        }
        h
    }
}

pub struct DecoderLayerT {
    self_attn: MultiHeadAttentionT,
    enc_dec_attn: MultiHeadAttentionT,
    ff: FeedForwardT,
}

impl DecoderLayerT {
    pub fn new(dim: usize, hidden: usize) -> Self {
        Self {
            self_attn: MultiHeadAttentionT::new(dim),
            enc_dec_attn: MultiHeadAttentionT::new(dim),
            ff: FeedForwardT::new(dim, hidden),
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
}

pub struct DecoderT {
    pub layers: Vec<DecoderLayerT>,
    pub embedding: EmbeddingT,
    pub proj: LinearT,
}

impl DecoderT {
    pub fn new(n: usize, vocab_size: usize, model_dim: usize, hidden: usize) -> Self {
        let mut v = Vec::new();
        for _ in 0..n {
            v.push(DecoderLayerT::new(model_dim, hidden));
        }
        Self {
            layers: v,
            embedding: EmbeddingT::new(vocab_size, model_dim),
            proj: LinearT::new(model_dim, vocab_size),
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
}

