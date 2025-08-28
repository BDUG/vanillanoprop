use crate::layers::Layer;
use crate::math::Matrix;
use crate::tensor::Tensor;

/// A simple container that applies layers sequentially.
pub struct Sequential {
    /// Ordered list of layers.
    pub layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    /// Create an empty sequential model.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Append a layer to the sequence.
    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    /// Forward pass used during inference.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out);
        }
        out
    }

    /// Forward pass used during training.
    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        let mut out = x.clone();
        for layer in self.layers.iter_mut() {
            out = layer.forward_train(&out);
        }
        out
    }

    /// Backward pass returning gradient with respect to the input.
    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let mut grad = grad_out.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
        grad
    }

    /// Zero any accumulated gradients in all layers.
    pub fn zero_grad(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.zero_grad();
        }
    }
}

