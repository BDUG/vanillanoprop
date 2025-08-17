use crate::autograd::Tensor;
use crate::linear_t::LinearT;

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
        let h = self.w1.forward(x);
        // ReLU
        let mut r = h.clone();
        if let Some(ref mut g) = r.grad {
            for v in g.data.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
        self.w2.forward(&r)
    }
}
