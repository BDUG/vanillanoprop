use crate::layers::Conv2d;
use crate::math::{self, Matrix};
use crate::rng::rng_from_env;
use rand::Rng;

struct DepthwiseSeparable {
    depthwise: Vec<Conv2d>,
    pointwise: Conv2d,
    stride: usize,
    in_channels: usize,
}

impl DepthwiseSeparable {
    fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let depthwise = (0..in_channels)
            .map(|_| Conv2d::new(1, 1, 3, stride, 1))
            .collect();
        let pointwise = Conv2d::new(in_channels, out_channels, 1, 1, 0);
        Self {
            depthwise,
            pointwise,
            stride,
            in_channels,
        }
    }

    fn forward(&mut self, x: &Matrix, h: usize) -> (Matrix, usize) {
        let out_h = (h + 2 * 1 - 3) / self.stride + 1;
        let mut depth_out = Matrix::zeros(1, self.in_channels * out_h * out_h);
        for c in 0..self.in_channels {
            let start = c * h * h;
            let end = start + h * h;
            let sub = Matrix::from_vec(1, h * h, x.data[start..end].to_vec());
            let conv_out = self.depthwise[c]
                .forward_local(&sub)
                .expect("invalid input to depthwise conv");
            let dst = c * out_h * out_h;
            depth_out.data[dst..dst + out_h * out_h].copy_from_slice(&conv_out.data);
        }
        let out = self
            .pointwise
            .forward_local(&depth_out)
            .expect("invalid input to pointwise conv");
        (out, out_h)
    }

    fn parameter_count(&self) -> usize {
        let mut count = 0usize;
        for dw in &self.depthwise {
            count += dw.w.w.data.len();
        }
        count += self.pointwise.w.w.data.len();
        count
    }
}

/// A minimal MobileNet-like convolutional network for 28x28 grayscale inputs.
///
/// The network uses a standard convolution followed by two depthwise-separable
/// convolution blocks and a final linear layer.
pub struct MobileNet {
    conv1: Conv2d,
    ds1: DepthwiseSeparable,
    ds2: DepthwiseSeparable,
    fc: Matrix,
    bias: Vec<f32>,
}

impl MobileNet {
    /// Create a new MobileNet with default channel sizes for MNIST.
    pub fn new(num_classes: usize) -> Self {
        let conv1 = Conv2d::new(1, 8, 3, 1, 1);
        let ds1 = DepthwiseSeparable::new(8, 16, 1);
        let ds2 = DepthwiseSeparable::new(16, 32, 2);
        let mut rng = rng_from_env();
        let mut w = Vec::with_capacity(32 * num_classes);
        for _ in 0..32 * num_classes {
            w.push(rng.gen_range(-0.01..0.01));
        }
        let fc = Matrix::from_vec(32, num_classes, w);
        let bias = vec![0.0; num_classes];
        Self {
            conv1,
            ds1,
            ds2,
            fc,
            bias,
        }
    }

    /// Forward pass returning the pooled feature vector and logits.
    pub fn forward(&mut self, img: &[u8]) -> (Vec<f32>, Vec<f32>) {
        let mut data = vec![0f32; 28 * 28];
        for (i, px) in img.iter().enumerate() {
            data[i] = *px as f32 / 255.0;
        }
        let mut x = Matrix::from_vec(1, 28 * 28, data);
        let mut h = 28usize;

        let mut out = self
            .conv1
            .forward_local(&x)
            .expect("invalid input to conv1");
        for v in out.data.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        x = out;

        let (mut out, mut h2) = self.ds1.forward(&x, h);
        for v in out.data.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        x = out;
        h = h2;

        let (mut out, h3) = self.ds2.forward(&x, h);
        for v in out.data.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        x = out;
        h2 = h3;

        // Global average pooling
        let channels = 32usize;
        let mut feat = vec![0f32; channels];
        for c in 0..channels {
            let start = c * h2 * h2;
            let end = start + h2 * h2;
            let sum: f32 = x.data[start..end].iter().sum();
            feat[c] = sum / (h2 * h2) as f32;
        }

        let mut logits = vec![0f32; self.fc.cols];
        for o in 0..self.fc.cols {
            let mut sum = self.bias[o];
            for i in 0..self.fc.rows {
                sum += feat[i] * self.fc.get(i, o);
            }
            logits[o] = sum;
        }
        math::inc_ops_by(self.fc.rows * self.fc.cols * 2);
        (feat, logits)
    }

    /// Batched forward pass returning feature and logit matrices.
    pub fn forward_batch(&mut self, imgs: &[Vec<u8>]) -> (Matrix, Matrix) {
        let bsz = imgs.len();
        let feat_dim = self.fc.rows;
        let mut feat = Matrix::zeros(bsz, feat_dim);
        let mut logits = Matrix::zeros(bsz, self.fc.cols);
        for (i, img) in imgs.iter().enumerate() {
            let (f, l) = self.forward(img);
            let fs = i * feat_dim;
            feat.data[fs..fs + feat_dim].copy_from_slice(&f);
            let ls = i * self.fc.cols;
            logits.data[ls..ls + self.fc.cols].copy_from_slice(&l);
        }
        (feat, logits)
    }

    /// Mutable access to the final classification layer parameters.
    pub fn parameters_mut(&mut self) -> (&mut Matrix, &mut Vec<f32>) {
        (&mut self.fc, &mut self.bias)
    }

    /// Immutable access to the final classification layer parameters.
    pub fn parameters(&self) -> (&Matrix, &Vec<f32>) {
        (&self.fc, &self.bias)
    }

    /// Total number of trainable parameters in the network.
    pub fn parameter_count(&self) -> usize {
        let mut count = self.conv1.w.w.data.len();
        count += self.ds1.parameter_count();
        count += self.ds2.parameter_count();
        count += self.fc.rows * self.fc.cols;
        count += self.bias.len();
        count
    }
}

