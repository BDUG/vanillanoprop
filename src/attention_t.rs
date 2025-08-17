use crate::autograd::Tensor;
use crate::linear_t::LinearT;

pub struct MultiHeadAttentionT {
    pub wq: LinearT,
    pub wk: LinearT,
    pub wv: LinearT,
    pub wo: LinearT,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl MultiHeadAttentionT {
    pub fn new(model_dim: usize, num_heads: usize) -> Self {
        let head_dim = model_dim / num_heads;
        Self {
            wq: LinearT::new(model_dim, model_dim),
            wk: LinearT::new(model_dim, model_dim),
            wv: LinearT::new(model_dim, model_dim),
            wo: LinearT::new(model_dim, model_dim),
            num_heads,
            head_dim,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);

        // ⚠️ stark vereinfachte Attention: keine Aufteilung in Köpfe,
        // kein Softmax, kein Masking – nur MatMul zur Demo
        let scores = Tensor::matmul(&q, &Tensor::matmul(&k, &v));
        self.wo.forward(&scores)
    }
}
