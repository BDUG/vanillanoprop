use crate::math::{self, Matrix};

/// Optimizer implementing the https://arxiv.org/abs/2506.21734 "Hierarchical Reasoning Model" algorithm.
///
/// The algorithm uses a simple decaying learning rate to update a linear layer
/// given the feature vector and gradient on the output logits.  This is merely
/// a lightweight placeholder capturing the core idea.
pub struct Hrm {
    pub lr: f32,
    pub decay: f32,
    step: usize,
}

impl Hrm {
    /// Create a new instance with the provided learning rate and decay factor.
    pub fn new(lr: f32, decay: f32) -> Self {
        Self { lr, decay, step: 0 }
    }

    /// Update the fully connected layer of a [`SimpleCNN`] using the algorithm
    /// described in the paper.
    pub fn update(&mut self, fc: &mut Matrix, bias: &mut [f32], grad: &[f32], feat: &[f32]) {
        self.step += 1;
        let lr = self.lr / (1.0 + self.decay * self.step as f32);
        let rows = fc.rows;
        let cols = fc.cols;
        for c in 0..cols {
            let g = grad[c];
            bias[c] -= lr * g;
            for r in 0..rows {
                let val = fc.get(r, c) - lr * g * feat[r];
                fc.set(r, c, val);
            }
        }
        let ops = cols * (2 + rows * 3);
        math::inc_ops_by(ops);
    }
}
