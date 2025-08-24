use crate::tensor::Tensor;
use crate::math::Matrix;
use super::linear::LinearT;

/// Common interface for network layers.
pub trait Layer {
    /// Forward pass used during inference.
    fn forward(&self, x: &Tensor) -> Tensor;

    /// Forward pass used during training, allowing the layer to cache values
    /// required for backward/feedback alignment updates.
    fn forward_train(&mut self, x: &Matrix) -> Matrix;

    /// Backward pass returning gradient with respect to the layer input.
    fn backward(&mut self, grad_out: &Matrix) -> Matrix;

    /// Zero any accumulated gradients.
    fn zero_grad(&mut self);

    /// Feedback-alignment style update. Returns the gradient with respect to the
    /// layer input.
    fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix;

    /// Perform an Adam optimisation step.
    fn adam_step(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    );

    /// Retrieve mutable references to parameters for optimisation/state
    /// serialisation.
    fn parameters(&mut self) -> Vec<&mut LinearT>;
}

