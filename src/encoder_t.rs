use crate::attention_t::MultiHeadAttentionT;
use crate::feedforward_t::FeedForwardT;
use crate::autograd::Tensor;
use crate::positional::positional_encoding;
use crate::math::Matrix;
use crate::embedding_t::EmbeddingT;

pub struct EncoderLayerT {
    pub attn: MultiHeadAttentionT,
    pub ff: FeedForwardT,
}

impl EncoderLayerT {
    pub fn new(dim: usize, heads: usize, hidden: usize) -> Self {
        Self {
            attn: MultiHeadAttentionT::new(dim, heads),
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
    pub fn new(n: usize, vocab_size: usize, model_dim: usize, heads: usize, hidden: usize) -> Self {
        let mut v = Vec::new();
        for _ in 0..n {
            v.push(EncoderLayerT::new(model_dim, heads, hidden));
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
