use crate::math::Matrix;
use crate::tensor::Tensor;
use super::layer::Layer;
use super::linear::LinearT;

/// Apply ReLU activation in place on a tensor.
pub fn forward_tensor(t: &mut Tensor) {
    for v in t.data.data.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Apply ReLU activation in place on a matrix and return a mask for backward.
pub fn forward_matrix(m: &mut Matrix) -> Vec<f32> {
    let mut mask = vec![0.0; m.data.len()];
    for (i, v) in m.data.iter_mut().enumerate() {
        if *v < 0.0 {
            *v = 0.0;
        } else {
            mask[i] = 1.0;
        }
    }
    mask
}

/// Apply the stored ReLU mask to the gradient matrix.
pub fn backward(grad: &mut Matrix, mask: &[f32]) {
    for (g, &m) in grad.data.iter_mut().zip(mask.iter()) {
        *g *= m;
    }
}

/// ReLU activation layer implementing the [`Layer`] trait.
pub struct ReLUT {
    mask: Vec<f32>,
}

impl ReLUT {
    /// Create a new ReLU layer.
    pub fn new() -> Self {
        Self { mask: Vec::new() }
    }

    fn forward_internal(x: &Tensor) -> Tensor {
        let mut out = x.clone();
        forward_tensor(&mut out);
        out
    }

    fn forward_train_internal(&mut self, x: &Matrix) -> Matrix {
        let mut out = x.clone();
        self.mask = forward_matrix(&mut out);
        out
    }

    fn backward_internal(&self, grad_out: &Matrix) -> Matrix {
        let mut grad = grad_out.clone();
        backward(&mut grad, &self.mask);
        grad
    }
}

impl Layer for ReLUT {
    fn forward(&self, x: &Tensor) -> Tensor {
        Self::forward_internal(x)
    }

    fn forward_train(&mut self, x: &Matrix) -> Matrix {
        self.forward_train_internal(x)
    }

    fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        self.backward_internal(grad_out)
    }

    fn zero_grad(&mut self) {}

    fn fa_update(&mut self, grad_out: &Matrix, _lr: f32) -> Matrix {
        self.backward_internal(grad_out)
    }

    fn adam_step(
        &mut self,
        _lr: f32,
        _beta1: f32,
        _beta2: f32,
        _eps: f32,
        _weight_decay: f32,
    ) {}

    fn parameters(&mut self) -> Vec<&mut LinearT> {
        Vec::new()
    }
}

