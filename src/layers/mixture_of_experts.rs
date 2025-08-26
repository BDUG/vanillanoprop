use super::layer::Layer;
use super::linear::LinearT;
use super::softmax::SoftmaxT;
use crate::math::Matrix;
use crate::tensor::Tensor;

/// Mixture of Experts layer with a simple gating network selecting between
/// multiple expert sub-networks.
pub struct MixtureOfExpertsT {
    /// Linear projection producing logits for each expert.
    pub gate: LinearT,
    /// Softmax turning logits into probabilities.
    pub softmax: SoftmaxT,
    /// Expert networks.  Each expert implements the `Layer` trait and is
    /// expected to take inputs of the same shape and produce outputs with the
    /// same dimensionality.
    pub experts: Vec<Box<dyn Layer>>,
    /// Number of experts to keep per input (top-k gating).  If equal to the
    /// number of experts, all experts are used.
    pub top_k: usize,
    // caches for backward/FA updates
    probs: Matrix,
    expert_outs: Vec<Matrix>,
}

impl MixtureOfExpertsT {
    /// Create a new mixture with `input_dim` inputs and the provided experts.
    /// `top_k` controls sparse routing; setting it to the number of experts
    /// disables sparsity.
    pub fn new(input_dim: usize, experts: Vec<Box<dyn Layer>>, top_k: usize) -> Self {
        let n = experts.len();
        Self {
            gate: LinearT::new(input_dim, n),
            softmax: SoftmaxT::new(),
            experts,
            top_k: top_k.min(n),
            probs: Matrix::zeros(0, 0),
            expert_outs: Vec::new(),
        }
    }

    fn mask_topk(&self, logits: &mut Matrix) {
        if self.top_k >= logits.cols { return; }
        for r in 0..logits.rows {
            let row_start = r * logits.cols;
            let mut indices: Vec<usize> = (0..logits.cols).collect();
            indices.sort_by(|&a, &b| {
                logits.data[row_start + b]
                    .partial_cmp(&logits.data[row_start + a])
                    .unwrap()
            });
            for &idx in indices[self.top_k..].iter() {
                logits.data[row_start + idx] = f32::NEG_INFINITY;
            }
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let num_exp = self.experts.len();
        let logits_t = self.gate.forward(x);
        let mut logits = logits_t.data.clone();
        self.mask_topk(&mut logits);
        let probs_t = self.softmax.forward(&Tensor::from_matrix(logits));
        let probs = probs_t.data;
        let first_out = self.experts[0].forward(x);
        let batch = first_out.data.rows;
        let dim = first_out.data.cols;
        let mut out = Matrix::zeros(batch, dim);
        for r in 0..batch {
            let g = probs.data[r * num_exp + 0];
            let row_start = r * dim;
            for c in 0..dim {
                out.data[row_start + c] += g * first_out.data.data[row_start + c];
            }
        }
        for (i, exp) in self.experts.iter().enumerate().skip(1) {
            let h = exp.forward(x);
            for r in 0..batch {
                let g = probs.data[r * num_exp + i];
                let row_start = r * dim;
                for c in 0..dim {
                    out.data[row_start + c] += g * h.data.data[row_start + c];
                }
            }
        }
        Tensor::from_matrix(out)
    }

    pub fn forward_local(&mut self, x: &Matrix) -> Matrix {
        let mut logits = self.gate.forward_local(x);
        self.mask_topk(&mut logits);
        self.probs = self.softmax.forward_train(&logits);
        self.expert_outs.clear();
        let first = self.experts[0].forward_train(x);
        let batch = first.rows;
        let dim = first.cols;
        let num_exp = self.experts.len();
        let mut out = Matrix::zeros(batch, dim);
        for r in 0..batch {
            let g = self.probs.data[r * num_exp + 0];
            let row_start = r * dim;
            for c in 0..dim {
                out.data[row_start + c] += g * first.data[row_start + c];
            }
        }
        self.expert_outs.push(first.clone());
        for (i, exp) in self.experts.iter_mut().enumerate().skip(1) {
            let h = exp.forward_train(x);
            for r in 0..batch {
                let g = self.probs.data[r * num_exp + i];
                let row_start = r * dim;
                for c in 0..dim {
                    out.data[row_start + c] += g * h.data[row_start + c];
                }
            }
            self.expert_outs.push(h);
        }
        out
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        self.forward_local(x)
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let batch = grad_out.rows;
        let dim = grad_out.cols;
        let num_exp = self.experts.len();
        let mut grad_input = Matrix::zeros(batch, self.gate.w.data.rows);
        let mut gate_grad = Matrix::zeros(batch, num_exp);
        for (i, exp) in self.experts.iter_mut().enumerate() {
            // gradient w.r.t gating probabilities
            for r in 0..batch {
                let row_start = r * dim;
                let mut dot = 0.0;
                for c in 0..dim {
                    dot += grad_out.data[row_start + c] * self.expert_outs[i].data[row_start + c];
                }
                gate_grad.data[r * num_exp + i] = dot;
            }
            // gradient propagated through expert
            let mut grad_exp = grad_out.clone();
            for r in 0..batch {
                let g = self.probs.data[r * num_exp + i];
                let row_start = r * dim;
                for c in 0..dim {
                    grad_exp.data[row_start + c] *= g;
                }
            }
            let grad_in = exp.backward(&grad_exp);
            grad_input = Matrix::add(&grad_input, &grad_in);
        }
        let grad_logits = self.softmax.backward(&gate_grad);
        let grad_gate_in = self.gate.backward(&grad_logits);
        Matrix::add(&grad_input, &grad_gate_in)
    }

    pub fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        let batch = grad_out.rows;
        let dim = grad_out.cols;
        let num_exp = self.experts.len();
        let mut grad_input = Matrix::zeros(batch, self.gate.w.data.rows);
        let mut gate_grad = Matrix::zeros(batch, num_exp);
        for (i, exp) in self.experts.iter_mut().enumerate() {
            for r in 0..batch {
                let row_start = r * dim;
                let mut dot = 0.0;
                for c in 0..dim {
                    dot += grad_out.data[row_start + c] * self.expert_outs[i].data[row_start + c];
                }
                gate_grad.data[r * num_exp + i] = dot;
            }
            let mut grad_exp = grad_out.clone();
            for r in 0..batch {
                let g = self.probs.data[r * num_exp + i];
                let row_start = r * dim;
                for c in 0..dim {
                    grad_exp.data[row_start + c] *= g;
                }
            }
            let grad_in = exp.fa_update(&grad_exp, lr);
            grad_input = Matrix::add(&grad_input, &grad_in);
        }
        let grad_logits = self.softmax.fa_update(&gate_grad, lr);
        let grad_gate_in = self.gate.fa_update(&grad_logits, lr);
        Matrix::add(&grad_input, &grad_gate_in)
    }

    pub fn zero_grad(&mut self) {
        self.gate.zero_grad();
        for e in self.experts.iter_mut() {
            e.zero_grad();
        }
    }

    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.gate.adam_step(lr, beta1, beta2, eps, weight_decay);
        for e in self.experts.iter_mut() {
            e.adam_step(lr, beta1, beta2, eps, weight_decay);
        }
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let mut params = self.gate.parameters();
        for e in self.experts.iter_mut() {
            params.extend(e.parameters());
        }
        params
    }
}

impl Layer for MixtureOfExpertsT {
    fn forward(&self, x: &Tensor) -> Tensor {
        MixtureOfExpertsT::forward(self, x)
    }

    fn forward_train(&mut self, x: &Matrix) -> Matrix {
        MixtureOfExpertsT::forward_train(self, x)
    }

    fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        MixtureOfExpertsT::backward(self, grad_out)
    }

    fn zero_grad(&mut self) {
        MixtureOfExpertsT::zero_grad(self);
    }

    fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        MixtureOfExpertsT::fa_update(self, grad_out, lr)
    }

    fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        MixtureOfExpertsT::adam_step(self, lr, beta1, beta2, eps, weight_decay);
    }

    fn parameters(&mut self) -> Vec<&mut LinearT> {
        MixtureOfExpertsT::parameters(self)
    }
}

