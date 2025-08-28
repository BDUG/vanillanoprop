pub mod adam;
pub mod hrm;
pub mod lr_scheduler;
pub mod sgd;

pub use adam::Adam;
pub use hrm::Hrm;
pub use lr_scheduler::{ConstantLr, CosineLr, LearningRateSchedule, LrScheduleConfig, StepLr};
pub use sgd::SGD;

use crate::layers::LinearT;
use crate::math::Matrix;
use std::any::Any;

/// Common interface for optimizers operating on linear layers.
pub trait Optimizer: Any {
    /// Update the provided parameters in-place.
    fn step(&mut self, params: &mut [&mut LinearT]);

    /// Optional update for raw weight matrices and bias vectors.
    ///
    /// Implementations can override this to handle cases where the model does
    /// not expose [`LinearT`] parameters directly.
    fn update_fc(&mut self, _fc: &mut Matrix, _bias: &mut [f32], _grad: &[f32], _feat: &[f32]) {}

    /// Allow downcasting to concrete types.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [&mut LinearT]) {
        Adam::step(self, params);
    }

    fn update_fc(&mut self, fc: &mut Matrix, bias: &mut [f32], grad: &[f32], feat: &[f32]) {
        // Basic gradient descent update using Adam's learning rate.
        let lr = self.lr;
        for (c, &g) in grad.iter().enumerate() {
            bias[c] -= lr * g;
            for (r, &f) in feat.iter().enumerate() {
                let idx = r * fc.cols + c;
                fc.data[idx] -= lr * f * g;
            }
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut [&mut LinearT]) {
        SGD::step(self, params);
    }

    fn update_fc(&mut self, fc: &mut Matrix, bias: &mut [f32], grad: &[f32], feat: &[f32]) {
        let lr = self.lr;
        for (c, &g) in grad.iter().enumerate() {
            bias[c] -= lr * g;
            for (r, &f) in feat.iter().enumerate() {
                let idx = r * fc.cols + c;
                fc.data[idx] -= lr * f * g;
            }
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Optimizer for Hrm {
    fn step(&mut self, _params: &mut [&mut LinearT]) {}

    fn update_fc(&mut self, fc: &mut Matrix, bias: &mut [f32], grad: &[f32], feat: &[f32]) {
        Hrm::update(self, fc, bias, grad, feat);
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Simple loss interface used during training.
pub trait Loss {
    /// Compute the scalar loss for the given prediction and target vectors.
    fn loss(&self, pred: &[f32], target: &[f32]) -> f32;
}

/// Mean squared error loss.
pub struct MeanSquaredError;

impl MeanSquaredError {
    pub fn new() -> Self {
        Self
    }
}

impl Loss for MeanSquaredError {
    fn loss(&self, pred: &[f32], target: &[f32]) -> f32 {
        let len = pred.len().min(target.len());
        if len == 0 {
            return 0.0;
        }
        let mut sum = 0.0f32;
        for i in 0..len {
            let d = pred[i] - target[i];
            sum += d * d;
        }
        sum / len as f32
    }
}

pub use MeanSquaredError as MseLoss;
