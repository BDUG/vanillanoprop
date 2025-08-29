use crate::math::Matrix;
use crate::tensor::Tensor;
use super::layer::Layer;
use super::linear::LinearT;

/// 2D max pooling forward pass.
///
/// `input` is a single feature map with dimensions `rows x cols`.
/// `kernel` is the pooling window size, `stride` is the step.
/// Returns the pooled output matrix and a vector of indices of the max
/// element for each pooling window. The indices are with respect to the
/// flattened input and can be used during backprop.
pub fn max_pool2d(input: &Matrix, kernel: usize, stride: usize) -> (Matrix, Vec<usize>) {
    let out_rows = (input.rows - kernel) / stride + 1;
    let out_cols = (input.cols - kernel) / stride + 1;
    let mut out = vec![0.0; out_rows * out_cols];
    let mut indices = vec![0usize; out_rows * out_cols];
    let mut idx = 0;
    for r in 0..out_rows {
        for c in 0..out_cols {
            let mut best = f32::NEG_INFINITY;
            let mut best_idx = 0usize;
            for kr in 0..kernel {
                for kc in 0..kernel {
                    let ir = r * stride + kr;
                    let ic = c * stride + kc;
                    let val = input.get(ir, ic);
                    if val > best {
                        best = val;
                        best_idx = ir * input.cols + ic;
                    }
                }
            }
            out[idx] = best;
            indices[idx] = best_idx;
            idx += 1;
        }
    }
    (Matrix::from_vec(out_rows, out_cols, out), indices)
}

/// Backward pass for 2D max pooling.
/// `grad` is the gradient with respect to the pooled output.
/// `indices` are the positions of maxima returned by `max_pool2d`.
pub fn max_pool2d_backward(
    grad: &Matrix,
    indices: &[usize],
    input_rows: usize,
    input_cols: usize,
) -> Matrix {
    let mut grad_input = Matrix::zeros(input_rows, input_cols);
    for (i, &idx) in indices.iter().enumerate() {
        grad_input.data[idx] += grad.data[i];
    }
    grad_input
}

/// 2D average pooling forward pass.
/// Returns pooled output and a mask containing the reciprocal of the
/// number of elements in each pooling region.
pub fn avg_pool2d(input: &Matrix, kernel: usize, stride: usize) -> (Matrix, Vec<f32>) {
    let out_rows = (input.rows - kernel) / stride + 1;
    let out_cols = (input.cols - kernel) / stride + 1;
    let mut out = vec![0.0; out_rows * out_cols];
    let mut mask = vec![0.0; out_rows * out_cols];
    let area = (kernel * kernel) as f32;
    let mut idx = 0;
    for r in 0..out_rows {
        for c in 0..out_cols {
            let mut sum = 0.0f32;
            for kr in 0..kernel {
                for kc in 0..kernel {
                    let ir = r * stride + kr;
                    let ic = c * stride + kc;
                    sum += input.get(ir, ic);
                }
            }
            out[idx] = sum / area;
            mask[idx] = 1.0 / area;
            idx += 1;
        }
    }
    (Matrix::from_vec(out_rows, out_cols, out), mask)
}

/// Backward pass for 2D average pooling using the mask from `avg_pool2d`.
pub fn avg_pool2d_backward(
    grad: &Matrix,
    mask: &[f32],
    input_rows: usize,
    input_cols: usize,
    kernel: usize,
    stride: usize,
) -> Matrix {
    let mut grad_input = Matrix::zeros(input_rows, input_cols);
    let mut idx = 0;
    let out_rows = grad.rows;
    let out_cols = grad.cols;
    for r in 0..out_rows {
        for c in 0..out_cols {
            let g = grad.get(r, c) * mask[idx];
            for kr in 0..kernel {
                for kc in 0..kernel {
                    let ir = r * stride + kr;
                    let ic = c * stride + kc;
                    grad_input.data[ir * input_cols + ic] += g;
                }
            }
            idx += 1;
        }
    }
    grad_input
}

/// Max pooling layer implementing the [`Layer`] trait.
pub struct MaxPool2d {
    kernel: usize,
    stride: usize,
    indices: Vec<usize>,
    input_rows: usize,
    input_cols: usize,
}

impl MaxPool2d {
    /// Create a new max pooling layer.
    pub fn new(kernel: usize, stride: usize) -> Self {
        Self {
            kernel,
            stride,
            indices: Vec::new(),
            input_rows: 0,
            input_cols: 0,
        }
    }

    /// Access the kernel size.
    pub fn kernel(&self) -> usize {
        self.kernel
    }

    /// Access the stride.
    pub fn stride(&self) -> usize {
        self.stride
    }

    fn forward_internal(&self, x: &Tensor) -> Tensor {
        let (out, _idx) = max_pool2d(&x.data, self.kernel, self.stride);
        Tensor::from_matrix(out)
    }

    fn forward_train_internal(&mut self, x: &Matrix) -> Matrix {
        self.input_rows = x.rows;
        self.input_cols = x.cols;
        let (out, idx) = max_pool2d(x, self.kernel, self.stride);
        self.indices = idx;
        out
    }

    fn backward_internal(&self, grad_out: &Matrix) -> Matrix {
        max_pool2d_backward(
            grad_out,
            &self.indices,
            self.input_rows,
            self.input_cols,
        )
    }
}

impl Layer for MaxPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_internal(x)
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
