use crate::layers::{LinearT, LSTM, GRU};
use crate::math::Matrix;
use crate::model::Model;
use crate::tensor::Tensor;

/// Reusable RNN encoder/decoder built from an LSTM or GRU cell.
///
/// The struct exposes a common interface so higher level code can train or
/// run inference without needing to know the specific recurrent cell used.
#[allow(clippy::upper_case_acronyms)]
pub enum RnnCell {
    /// Long Short-Term Memory cell
    LSTM(LSTM),
    /// Gated Recurrent Unit cell
    GRU(GRU),
}

/// Simple RNN model composed of a recurrent cell followed by a linear
/// projection to the desired output dimension.
pub struct RNN {
    pub cell: RnnCell,
    pub fc: LinearT,
    last_seq_len: usize,
}

impl RNN {
    /// Create a new RNN using an LSTM cell.
    pub fn new_lstm(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            cell: RnnCell::LSTM(LSTM::new(input_dim, hidden_dim)),
            fc: LinearT::new(hidden_dim, output_dim),
            last_seq_len: 0,
        }
    }

    /// Create a new RNN using a GRU cell.
    pub fn new_gru(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            cell: RnnCell::GRU(GRU::new(input_dim, hidden_dim)),
            fc: LinearT::new(hidden_dim, output_dim),
            last_seq_len: 0,
        }
    }

    /// Forward pass for inference. `x` is a sequence represented as a matrix
    /// where each row is a time step.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = match &self.cell {
            RnnCell::LSTM(l) => l.forward(x),
            RnnCell::GRU(g) => g.forward(x),
        };
        // take last hidden state
        let last_row = h.data.rows - 1;
        let mut last = Matrix::zeros(1, h.data.cols);
        for c in 0..h.data.cols {
            last.set(0, c, h.data.get(last_row, c));
        }
        self.fc.forward(&Tensor::from_matrix(last))
    }

    /// Training forward pass.
    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        let h = match &mut self.cell {
            RnnCell::LSTM(l) => l.forward_train(x),
            RnnCell::GRU(g) => g.forward_train(x),
        };
        self.last_seq_len = h.rows;
        let last = Matrix::from_vec(1, h.cols, h.data[(h.rows - 1) * h.cols..].to_vec());
        self.fc.forward_train(&last)
    }

    /// Backward pass through the linear projection and the recurrent cell.
    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let grad_last = self.fc.backward(grad_out);
        let hidden_dim = grad_last.cols;
        let mut grad_seq = Matrix::zeros(self.last_seq_len, hidden_dim);
        for c in 0..hidden_dim {
            grad_seq.set(self.last_seq_len - 1, c, grad_last.get(0, c));
        }
        match &mut self.cell {
            RnnCell::LSTM(l) => l.backward(&grad_seq),
            RnnCell::GRU(g) => g.backward(&grad_seq),
        }
    }

    pub fn zero_grad(&mut self) {
        match &mut self.cell {
            RnnCell::LSTM(l) => l.zero_grad(),
            RnnCell::GRU(g) => g.zero_grad(),
        }
        self.fc.zero_grad();
    }

    /// Feedback alignment style update.
    pub fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        let grad_last = self.fc.fa_update(grad_out, lr);
        let hidden_dim = grad_last.cols;
        let mut grad_seq = Matrix::zeros(self.last_seq_len, hidden_dim);
        for c in 0..hidden_dim {
            grad_seq.set(self.last_seq_len - 1, c, grad_last.get(0, c));
        }
        match &mut self.cell {
            RnnCell::LSTM(l) => l.fa_update(&grad_seq, lr),
            RnnCell::GRU(g) => g.fa_update(&grad_seq, lr),
        }
    }

    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        match &mut self.cell {
            RnnCell::LSTM(l) => l.adam_step(lr, beta1, beta2, eps, weight_decay),
            RnnCell::GRU(g) => g.adam_step(lr, beta1, beta2, eps, weight_decay),
        }
        self.fc.adam_step(lr, beta1, beta2, eps, weight_decay);
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let mut params = match &mut self.cell {
            RnnCell::LSTM(l) => l.parameters(),
            RnnCell::GRU(g) => g.parameters(),
        };
        params.extend(self.fc.parameters());
        params
    }
}

/// Build a generic RNN architecture as a [`Model`] graph.
/// `cell` should describe the recurrent cell used (e.g. "LSTM" or "GRU").
pub fn rnn_model(cell: &str) -> Model {
    let mut m = Model::new();
    let input = m.add("input_seq");
    let cell_node = m.add(format!("{}", cell));
    let fc = m.add("fc");
    m.connect(input, cell_node);
    m.connect(cell_node, fc);
    m
}
