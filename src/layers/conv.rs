use super::layer::Layer;
use super::linear::LinearT;
use crate::math::Matrix;
use crate::tensor::Tensor;
use std::fmt;

/// 2D convolution layer using im2col and a linear weight matrix.
///
/// The implementation is intentionally simple and only supports
/// square inputs. Each input is assumed to have shape
/// `(batch, in_channels * height * width)` where `height == width`.
/// The layer performs a standard 2D convolution with the given
/// kernel size, stride and padding. The weights are stored in a
/// [`LinearT`] allowing reuse of the existing optimisation code.
pub struct Conv2d {
    pub w: LinearT,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    // Cached shapes from the last forward pass required for backward.
    last_input_shape: (usize, usize, usize), // (batch, in_h, in_w)
    last_output_shape: (usize, usize),       // (out_h, out_w)
}

#[derive(Debug, PartialEq)]
pub enum ConvError {
    ChannelMismatch { features: usize, in_channels: usize },
    NonSquareInput { size: usize },
}

impl fmt::Display for ConvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConvError::ChannelMismatch {
                features,
                in_channels,
            } => write!(
                f,
                "Input feature count {} is not divisible by in_channels {}",
                features, in_channels
            ),
            ConvError::NonSquareInput { size } => {
                write!(f, "Input spatial size {} is not a perfect square", size)
            }
        }
    }
}

impl std::error::Error for ConvError {}

impl Conv2d {
    /// Create a new convolution layer.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let in_dim = in_channels * kernel_size * kernel_size;
        let w = LinearT::new(in_dim, out_channels);
        Self {
            w,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            last_input_shape: (0, 0, 0),
            last_output_shape: (0, 0),
        }
    }

    fn compute_shapes(&self, x: &Matrix) -> Result<(usize, usize, usize, usize, usize), ConvError> {
        let batch = x.rows;
        if x.cols % self.in_channels != 0 {
            return Err(ConvError::ChannelMismatch {
                features: x.cols,
                in_channels: self.in_channels,
            });
        }
        let in_hw = x.cols / self.in_channels;
        let in_h = (in_hw as f32).sqrt() as usize;
        if in_h * in_h != in_hw {
            return Err(ConvError::NonSquareInput { size: in_hw });
        }
        let in_w = in_h; // assume square inputs
        let out_h = (in_h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (in_w + 2 * self.padding - self.kernel_size) / self.stride + 1;
        Ok((batch, in_h, in_w, out_h, out_w))
    }

    fn im2col(&self, x: &Matrix, in_h: usize, in_w: usize, out_h: usize, out_w: usize) -> Matrix {
        let batch = x.rows;
        let mut cols = Matrix::zeros(
            batch * out_h * out_w,
            self.in_channels * self.kernel_size * self.kernel_size,
        );
        let mut row = 0;
        for b in 0..batch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut col_idx = 0;
                    for ic in 0..self.in_channels {
                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let ih = oh * self.stride + kh as usize;
                                let iw = ow * self.stride + kw as usize;
                                let ihp = ih as isize - self.padding as isize;
                                let iwp = iw as isize - self.padding as isize;
                                let val = if ihp >= 0
                                    && ihp < in_h as isize
                                    && iwp >= 0
                                    && iwp < in_w as isize
                                {
                                    let idx = b * x.cols
                                        + ic * in_h * in_w
                                        + ihp as usize * in_w
                                        + iwp as usize;
                                    x.data[idx]
                                } else {
                                    0.0
                                };
                                cols.set(row, col_idx, val);
                                col_idx += 1;
                            }
                        }
                    }
                    row += 1;
                }
            }
        }
        cols
    }

    fn col2im(
        &self,
        cols: &Matrix,
        batch: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
    ) -> Matrix {
        let mut img = Matrix::zeros(batch, self.in_channels * in_h * in_w);
        let mut row = 0;
        for b in 0..batch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut col_idx = 0;
                    for ic in 0..self.in_channels {
                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let ih = oh * self.stride + kh as usize;
                                let iw = ow * self.stride + kw as usize;
                                let ihp = ih as isize - self.padding as isize;
                                let iwp = iw as isize - self.padding as isize;
                                if ihp >= 0
                                    && ihp < in_h as isize
                                    && iwp >= 0
                                    && iwp < in_w as isize
                                {
                                    let val = cols.get(row, col_idx);
                                    let idx = b * img.cols
                                        + ic * in_h * in_w
                                        + ihp as usize * in_w
                                        + iwp as usize;
                                    img.data[idx] += val;
                                }
                                col_idx += 1;
                            }
                        }
                    }
                    row += 1;
                }
            }
        }
        img
    }

    fn reshape_output(
        &self,
        out_cols: &Matrix,
        batch: usize,
        out_h: usize,
        out_w: usize,
    ) -> Matrix {
        let mut out = Matrix::zeros(batch, self.out_channels * out_h * out_w);
        let mut row = 0;
        for b in 0..batch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for oc in 0..self.out_channels {
                        let val = out_cols.get(row, oc);
                        let idx = oc * out_h * out_w + oh * out_w + ow;
                        out.set(b, idx, val);
                    }
                    row += 1;
                }
            }
        }
        out
    }

    pub fn forward_local(&mut self, x: &Matrix) -> Result<Matrix, ConvError> {
        let (batch, in_h, in_w, out_h, out_w) = self.compute_shapes(x)?;
        let cols = self.im2col(x, in_h, in_w, out_h, out_w);
        let out_cols = self.w.forward_local(&cols);
        self.last_input_shape = (batch, in_h, in_w);
        self.last_output_shape = (out_h, out_w);
        Ok(self.reshape_output(&out_cols, batch, out_h, out_w))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, ConvError> {
        let (batch, in_h, in_w, out_h, out_w) = self.compute_shapes(&x.data)?;
        let cols = self.im2col(&x.data, in_h, in_w, out_h, out_w);
        let out_cols = self.w.forward(&Tensor::from_matrix(cols));
        let out = self.reshape_output(&out_cols.data, batch, out_h, out_w);
        Ok(Tensor::from_matrix(out))
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let (batch, in_h, in_w) = self.last_input_shape;
        let (out_h, out_w) = self.last_output_shape;
        let mut grad_cols = Matrix::zeros(batch * out_h * out_w, self.out_channels);
        let mut row = 0;
        for b in 0..batch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for oc in 0..self.out_channels {
                        let idx = oc * out_h * out_w + oh * out_w + ow;
                        let val = grad_out.get(b, idx);
                        grad_cols.set(row, oc, val);
                    }
                    row += 1;
                }
            }
        }
        let grad_in_cols = self.w.backward(&grad_cols);
        self.col2im(&grad_in_cols, batch, in_h, in_w, out_h, out_w)
    }

    pub fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        let (batch, in_h, in_w) = self.last_input_shape;
        let (out_h, out_w) = self.last_output_shape;
        let mut grad_cols = Matrix::zeros(batch * out_h * out_w, self.out_channels);
        let mut row = 0;
        for b in 0..batch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for oc in 0..self.out_channels {
                        let idx = oc * out_h * out_w + oh * out_w + ow;
                        let val = grad_out.get(b, idx);
                        grad_cols.set(row, oc, val);
                    }
                    row += 1;
                }
            }
        }
        let grad_in_cols = self.w.fa_update(&grad_cols, lr);
        self.col2im(&grad_in_cols, batch, in_h, in_w, out_h, out_w)
    }

    pub fn zero_grad(&mut self) {
        self.w.zero_grad();
    }

    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.w.adam_step(lr, beta1, beta2, eps, weight_decay);
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let w = &mut self.w;
        vec![w]
    }

    /// Accessor methods for exporting or inspection.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn padding(&self) -> usize {
        self.padding
    }
}

impl Layer for Conv2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        Conv2d::forward(self, x).expect("invalid input to Conv2d forward")
    }

    fn forward_train(&mut self, x: &Matrix) -> Matrix {
        Conv2d::forward_local(self, x).expect("invalid input to Conv2d forward_local")
    }

    fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        Conv2d::backward(self, grad_out)
    }

    fn zero_grad(&mut self) {
        Conv2d::zero_grad(self);
    }

    fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        Conv2d::fa_update(self, grad_out, lr)
    }

    fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        Conv2d::adam_step(self, lr, beta1, beta2, eps, weight_decay);
    }

    fn parameters(&mut self) -> Vec<&mut LinearT> {
        Conv2d::parameters(self)
    }
}
