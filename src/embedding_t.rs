use crate::autograd::Tensor;
use crate::linear_t::LinearT;

/// Embedding layer: maps one-hot (vocab_size) into dense model_dim.
pub struct EmbeddingT {
    pub table: LinearT, // weight matrix (vocab_size x model_dim)
}

impl EmbeddingT {
    pub fn new(vocab_size: usize, model_dim: usize) -> Self {
        Self { table: LinearT::new(vocab_size, model_dim) }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x: (seq_len x vocab_size) → multiply with (vocab_size x model_dim)
        Tensor::matmul(x, &self.table.w)
    }
}
