use super::layer::Layer;
use super::linear::LinearT;
use super::softmax::SoftmaxT;
use crate::math::Matrix;
use crate::tensor::Tensor;
use std::cmp::Ordering;

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
            indices.select_nth_unstable_by(self.top_k, |&a, &b| {
                logits.data[row_start + b]
                    .partial_cmp(&logits.data[row_start + a])
                    .unwrap_or(Ordering::Equal)
            });
            for &idx in indices[self.top_k..].iter() {
                logits.data[row_start + idx] = f32::NEG_INFINITY;
            }
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let num_exp = self.experts.len();
        let logits_t = self.gate.forward(x);
        let mut logits =
            Matrix::from_vec(logits_t.shape[0], logits_t.shape[1], logits_t.data.clone());
        self.mask_topk(&mut logits);
        let probs_t = self.softmax.forward(&Tensor::from_matrix(logits.clone()));
        let probs =
            Matrix::from_vec(probs_t.shape[0], probs_t.shape[1], probs_t.data.clone());

        // Compute all expert outputs first so we can combine them uniformly
        let expert_outs: Vec<Matrix> = self
            .experts
            .iter()
            .map(|e| {
                let t = e.forward(x);
                Matrix::from_vec(t.shape[0], t.shape[1], t.data)
            })
            .collect();
        let batch = expert_outs[0].rows;
        let dim = expert_outs[0].cols;
        let mut out = Matrix::zeros(batch, dim);

        for (i, h) in expert_outs.iter().enumerate() {
            for (prob_row, (out_row, h_row)) in probs
                .data
                .chunks(num_exp)
                .zip(out.data.chunks_mut(dim).zip(h.data.chunks(dim)))
            {
                let g = prob_row[i];
                for c in 0..dim {
                    out_row[c] += g * h_row[c];
                }
            }
        }

        Tensor::from_matrix(out)
    }

    pub fn forward_local(&mut self, x: &Matrix) -> Matrix {
        let mut logits = self.gate.forward_local(x);
        self.mask_topk(&mut logits);

        // cache probabilities, resizing if necessary
        let probs = self.softmax.forward_train(&logits);
        if self.probs.rows != probs.rows || self.probs.cols != probs.cols {
            self.probs = Matrix::zeros(probs.rows, probs.cols);
        }
        self.probs.data.clone_from_slice(&probs.data);
        self.probs.rows = probs.rows;
        self.probs.cols = probs.cols;

        let num_exp = self.experts.len();
        // ensure expert_outs has space for all experts
        if self.expert_outs.len() != num_exp {
            self.expert_outs = vec![Matrix::zeros(0, 0); num_exp];
        }
        for (slot, exp) in self.expert_outs.iter_mut().zip(self.experts.iter_mut()) {
            *slot = exp.forward_train(x);
        }

        let batch = self.expert_outs[0].rows;
        let dim = self.expert_outs[0].cols;
        let mut out = Matrix::zeros(batch, dim);

        for (i, h) in self.expert_outs.iter().enumerate() {
            for (prob_row, (out_row, h_row)) in self
                .probs
                .data
                .chunks(num_exp)
                .zip(out.data.chunks_mut(dim).zip(h.data.chunks(dim)))
            {
                let g = prob_row[i];
                for c in 0..dim {
                    out_row[c] += g * h_row[c];
                }
            }
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
        let mut grad_input = Matrix::zeros(batch, self.gate.w.shape[0]);
        let mut gate_grad = Matrix::zeros(batch, num_exp);
        let mut grad_exp = Matrix::zeros(batch, dim);

        for (i, (exp, h)) in self.experts.iter_mut().zip(self.expert_outs.iter()).enumerate() {
            grad_exp.data.clone_from_slice(&grad_out.data);
            for (r, ((prob_row, grad_row), h_row)) in self
                .probs
                .data
                .chunks(num_exp)
                .zip(grad_out.data.chunks(dim))
                .zip(h.data.chunks(dim))
                .enumerate()
            {
                let g = prob_row[i];
                let mut dot = 0.0;
                let row_start = r * dim;
                for c in 0..dim {
                    dot += grad_row[c] * h_row[c];
                    grad_exp.data[row_start + c] = g * grad_row[c];
                }
                gate_grad.data[r * num_exp + i] = dot;
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
        let mut grad_input = Matrix::zeros(batch, self.gate.w.shape[0]);
        let mut gate_grad = Matrix::zeros(batch, num_exp);
        let mut grad_exp = Matrix::zeros(batch, dim);

        for (i, (exp, h)) in self.experts.iter_mut().zip(self.expert_outs.iter()).enumerate() {
            grad_exp.data.clone_from_slice(&grad_out.data);
            for (r, ((prob_row, grad_row), h_row)) in self
                .probs
                .data
                .chunks(num_exp)
                .zip(grad_out.data.chunks(dim))
                .zip(h.data.chunks(dim))
                .enumerate()
            {
                let g = prob_row[i];
                let mut dot = 0.0;
                let row_start = r * dim;
                for c in 0..dim {
                    dot += grad_row[c] * h_row[c];
                    grad_exp.data[row_start + c] = g * grad_row[c];
                }
                gate_grad.data[r * num_exp + i] = dot;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn build_moe(top_k: usize) -> MixtureOfExpertsT {
        MixtureOfExpertsT {
            gate: LinearT::new(0, 0),
            softmax: SoftmaxT::new(),
            experts: vec![],
            top_k,
            probs: Matrix::zeros(0, 0),
            expert_outs: Vec::new(),
        }
    }

    #[test]
    fn masks_all_but_topk() {
        let moe = build_moe(2);
        let mut logits = Matrix::from_vec(1, 5, vec![1.0, 3.0, 2.0, 5.0, 4.0]);
        moe.mask_topk(&mut logits);
        let row = &logits.data;
        assert!(row[0].is_infinite() && row[0].is_sign_negative());
        assert!(row[1].is_infinite() && row[1].is_sign_negative());
        assert!(row[2].is_infinite() && row[2].is_sign_negative());
        assert_eq!(row[3], 5.0);
        assert_eq!(row[4], 4.0);
    }

    #[test]
    fn no_mask_when_topk_covers_all() {
        let moe = build_moe(5);
        let mut logits = Matrix::from_vec(1, 5, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        moe.mask_topk(&mut logits);
        for &v in &logits.data {
            assert!(!v.is_infinite());
        }
    }
}

