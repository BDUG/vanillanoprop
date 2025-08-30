use crate::layers::{Activation, LinearT};
use crate::math::Matrix;
use crate::models::{TransformerEncoderLayer, RNN};
use crate::tensor::Tensor;

/// Hybrid sequence model that applies an RNN followed by a Transformer encoder layer.
pub struct HybridRnnTransformer {
    /// Recurrent encoder applied first.
    pub rnn: RNN,
    /// Transformer layer applied to the RNN output.
    pub transformer: TransformerEncoderLayer,
}

impl HybridRnnTransformer {
    /// Build a new hybrid model.
    ///
    /// `input_dim` - dimensionality of the input sequence at each time step.
    /// `hidden_dim` - hidden size of the RNN cell.
    /// `model_dim` - dimensionality of the features passed between the RNN and Transformer.
    /// `num_heads` - number of attention heads in the Transformer layer.
    /// `ff_hidden` - hidden size of the Transformer's feed forward network.
    /// `p_drop` - dropout probability used by the Transformer layer.
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        model_dim: usize,
        num_heads: usize,
        ff_hidden: usize,
        p_drop: f32,
    ) -> Self {
        let rnn = RNN::new_lstm(input_dim, hidden_dim, model_dim);
        let transformer =
            TransformerEncoderLayer::new(model_dim, num_heads, ff_hidden, Activation::ReLU, p_drop);
        Self { rnn, transformer }
    }

    /// Inference forward pass through RNN then Transformer layer.
    pub fn forward(&mut self, x: &Tensor, mask: Option<&Matrix>) -> Tensor {
        let rnn_out = self.rnn.forward(x);
        self.transformer.forward(&rnn_out, mask)
    }

    /// Training forward pass.
    pub fn forward_train(&mut self, x: &Matrix, mask: Option<&Matrix>) -> Matrix {
        let rnn_out = self.rnn.forward_train(x);
        self.transformer.forward_train(&rnn_out, mask)
    }

    /// Backward pass through Transformer then RNN.
    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let g = self.transformer.backward(grad_out);
        self.rnn.backward(&g)
    }

    /// Reset gradients of both sub modules.
    pub fn zero_grad(&mut self) {
        self.rnn.zero_grad();
        self.transformer.zero_grad();
    }

    /// Perform an Adam optimisation step on all parameters.
    pub fn adam_step(&mut self, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32) {
        self.rnn.adam_step(lr, b1, b2, eps, wd);
        self.transformer.adam_step(lr, b1, b2, eps, wd);
    }

    /// Gather all trainable parameters from both modules.
    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let mut params = self.rnn.parameters();
        params.extend(self.transformer.parameters());
        params
    }
}

/// Helper to build a [`Model`] graph representing a hybrid RNN + Transformer block.
pub fn hybrid_rnn_transformer_model() -> crate::model::Model {
    use crate::model::Model;
    let mut m = Model::new();
    let input = m.add("input_seq");
    let rnn = m.add("rnn");
    let transformer = m.add("transformer");
    m.connect(input, rnn);
    m.connect(rnn, transformer);
    m
}
