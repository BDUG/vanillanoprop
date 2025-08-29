use crate::layers::{max_pool2d, Conv2d};
use crate::math::{self, Matrix};
use crate::model::Model;
use crate::rng::rng_from_env;
use rand::Rng;

/// A minimal VGG style convolutional network.
///
/// The network is composed of blocks of convolution + ReLU layers followed by
/// a 2x2 max pooling operation. The number of convolution layers in each block
/// is configured by the `blocks` argument allowing construction of common
/// variants such as VGG16 (`[2, 2, 3, 3, 3]`) or VGG19
/// (`[2, 2, 4, 4, 4]`). After the convolutional trunk a single linear layer is
/// used as the classification head.
pub struct VGG {
    convs: Vec<Conv2d>,
    fc: Matrix,
    bias: Vec<f32>,
    blocks: Vec<usize>,
    channels: Vec<usize>,
}

impl VGG {
    /// Create a new VGG network for grayscale 28x28 inputs.
    ///
    /// `blocks` specifies the number of convolution layers in each block and
    /// `num_classes` controls the size of the final classification layer.
    pub fn new(blocks: &[usize], num_classes: usize) -> Self {
        let mut convs = Vec::new();
        let mut channels = Vec::new();
        let mut in_channels = 1usize; // MNIST is single channel
        let mut h = 28usize;
        for (i, &num_layers) in blocks.iter().enumerate() {
            let out_channels = 8 << i; // keep the network small
            for _ in 0..num_layers {
                convs.push(Conv2d::new(in_channels, out_channels, 3, 1, 1));
                in_channels = out_channels;
            }
            channels.push(out_channels);
            // spatial size after 2x2 max pooling
            h = (h - 2) / 2 + 1;
        }
        let feature_dim = in_channels * h * h;
        let mut rng = rng_from_env();
        let mut w = Vec::with_capacity(feature_dim * num_classes);
        for _ in 0..feature_dim * num_classes {
            w.push(rng.gen_range(-0.01..0.01));
        }
        let fc = Matrix::from_vec(feature_dim, num_classes, w);
        let bias = vec![0.0; num_classes];
        Self { convs, fc, bias, blocks: blocks.to_vec(), channels }
    }

    /// Forward pass returning the flattened feature map and logits.
    pub fn forward(&mut self, img: &[u8]) -> (Vec<f32>, Vec<f32>) {
        let mut h = 28usize;
        let mut data = vec![0f32; 28 * 28];
        for (i, px) in img.iter().enumerate() {
            data[i] = *px as f32 / 255.0;
        }
        let mut x = Matrix::from_vec(1, 28 * 28, data);
        let mut conv_idx = 0;
        for (b, &num_layers) in self.blocks.iter().enumerate() {
            for _ in 0..num_layers {
                let conv = &mut self.convs[conv_idx];
                let mut out = conv
                    .forward_local(&x)
                    .expect("invalid input to convolution");
                for v in out.data.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0; // ReLU
                    }
                }
                x = out;
                conv_idx += 1;
            }
            let out_channels = self.channels[b];
            let new_h = (h - 2) / 2 + 1;
            let mut pooled = Vec::with_capacity(out_channels * new_h * new_h);
            for c in 0..out_channels {
                let start = c * h * h;
                let end = start + h * h;
                let fm = Matrix::from_vec(h, h, x.data[start..end].to_vec());
                let (p, _) = max_pool2d(&fm, 2, 2);
                pooled.extend_from_slice(&p.data);
            }
            x = Matrix::from_vec(1, out_channels * new_h * new_h, pooled);
            h = new_h;
        }
        let feat = x.data.clone();
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

    /// Mutable access to the final classification layer parameters.
    pub fn parameters_mut(&mut self) -> (&mut Matrix, &mut Vec<f32>) {
        (&mut self.fc, &mut self.bias)
    }
}

/// Build a VGG style architecture as a [`Model`] graph.
///
/// `blocks` specifies the number of convolution layers in each block.
pub fn vgg_model(blocks: &[usize], num_classes: usize) -> Model {
    let mut m = Model::new();
    let input = m.add("input");
    let mut prev = input;
    for (b, &num_layers) in blocks.iter().enumerate() {
        for l in 0..num_layers {
            let conv = m.add(format!("block{}_conv{}", b, l));
            let relu = m.add(format!("block{}_relu{}", b, l));
            m.connect(prev, conv);
            m.connect(conv, relu);
            prev = relu;
        }
        let pool = m.add(format!("block{}_pool", b));
        m.connect(prev, pool);
        prev = pool;
    }
    let flatten = m.add("flatten");
    m.connect(prev, flatten);
    let fc = m.add(format!("fc{}", num_classes));
    m.connect(flatten, fc);
    m
}
