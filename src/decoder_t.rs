use crate::attention_t::MultiHeadAttentionT;
use crate::feedforward_t::FeedForwardT;
use crate::autograd::Tensor;
use crate::embedding_t::EmbeddingT;

pub struct DecoderLayerT {
    self_attn: MultiHeadAttentionT,
    enc_dec_attn: MultiHeadAttentionT,
    ff: FeedForwardT,
}

impl DecoderLayerT {
    pub fn new(dim: usize, heads: usize, hidden: usize) -> Self {
        Self {
            self_attn: MultiHeadAttentionT::new(dim, heads),
            enc_dec_attn: MultiHeadAttentionT::new(dim, heads),
            ff: FeedForwardT::new(dim, hidden),
        }
    }

    pub fn forward(&self, x: &Tensor, enc_out: &Tensor) -> Tensor {
        let h1 = self.self_attn.forward(x);
        let h2 = self.enc_dec_attn.forward(&Tensor::add(&h1, enc_out));
        self.ff.forward(&h2)
    }
}

pub struct DecoderT {
    pub layers: Vec<DecoderLayerT>,
    pub embedding: EmbeddingT,
}

impl DecoderT {
    pub fn new(n: usize, vocab_size: usize, model_dim: usize, heads: usize, hidden: usize) -> Self {
        let mut v = Vec::new();
        for _ in 0..n {
            v.push(DecoderLayerT::new(model_dim, heads, hidden));
        }
        Self {
            layers: v,
            embedding: EmbeddingT::new(vocab_size, model_dim),
        }
    }

    pub fn forward(&self, one_hot_x: &Tensor, enc_out: &Tensor) -> Tensor {
        // one_hot_x = (seq_len × vocab_size)
        let mut h = self.embedding.forward(one_hot_x);
        for l in &self.layers {
            h = l.forward(&h, enc_out);
        }
        h
    }
}
