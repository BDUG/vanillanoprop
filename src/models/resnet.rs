use crate::math::{self, Matrix};
use crate::model::Model;
use crate::rng::rng_from_env;
use rand::Rng;

struct ResidualBlock {
    w1: Matrix,
    b1: Vec<f32>,
    w2: Matrix,
    b2: Vec<f32>,
}

/// A very small fully connected ResNet-like model.
///
/// The network projects the 28x28 input image into a hidden dimension
/// and applies a sequence of residual blocks before a final linear layer.
/// The number of residual blocks and the hidden dimension are configurable.
pub struct ResNet {
    input: Matrix,
    input_bias: Vec<f32>,
    blocks: Vec<ResidualBlock>,
    fc: Matrix,
    bias: Vec<f32>,
}

impl ResidualBlock {
    fn new(depth: usize, rng: &mut impl Rng) -> Self {
        let mut w1 = vec![0.0; depth * depth];
        let mut w2 = vec![0.0; depth * depth];
        for w in w1.iter_mut().chain(w2.iter_mut()) {
            *w = rng.gen_range(-0.01..0.01);
        }
        let b1 = vec![0.0; depth];
        let b2 = vec![0.0; depth];
        Self {
            w1: Matrix::from_vec(depth, depth, w1),
            b1,
            w2: Matrix::from_vec(depth, depth, w2),
            b2,
        }
    }
}

impl ResNet {
    /// Create a new [`ResNet`].
    ///
    /// `num_classes` controls the number of output classes.
    /// `depth` sets the hidden dimension of the residual blocks.
    /// `num_blocks` configures how many residual blocks are stacked.
    pub fn new(num_classes: usize, depth: usize, num_blocks: usize) -> Self {
        let mut rng = rng_from_env();
        // Project the 28x28 input to the hidden dimension.
        let mut w_in = vec![0.0; 28 * 28 * depth];
        for w in &mut w_in {
            *w = rng.gen_range(-0.01..0.01);
        }
        let input = Matrix::from_vec(28 * 28, depth, w_in);
        let input_bias = vec![0.0; depth];

        let mut blocks = Vec::new();
        for _ in 0..num_blocks {
            blocks.push(ResidualBlock::new(depth, &mut rng));
        }

        // Final classification layer.
        let mut w_out = vec![0.0; depth * num_classes];
        for w in &mut w_out {
            *w = rng.gen_range(-0.01..0.01);
        }
        let fc = Matrix::from_vec(depth, num_classes, w_out);
        let bias = vec![0.0; num_classes];

        Self {
            input,
            input_bias,
            blocks,
            fc,
            bias,
        }
    }

    /// Forward pass returning the hidden feature vector and logits.
    pub fn forward(&self, img: &[u8]) -> (Vec<f32>, Vec<f32>) {
        let mut x = vec![0.0f32; self.input.rows];
        for (i, v) in x.iter_mut().enumerate() {
            *v = img[i] as f32 / 255.0;
        }

        // Input projection.
        let mut h = vec![0.0f32; self.input.cols];
        for o in 0..self.input.cols {
            let mut sum = self.input_bias[o];
            for i in 0..self.input.rows {
                sum += x[i] * self.input.get(i, o);
            }
            h[o] = if sum > 0.0 { sum } else { 0.0 };
        }

        // Residual blocks.
        for blk in &self.blocks {
            let mut z1 = vec![0.0f32; h.len()];
            for o in 0..h.len() {
                let mut sum = blk.b1[o];
                for i in 0..h.len() {
                    sum += h[i] * blk.w1.get(i, o);
                }
                z1[o] = if sum > 0.0 { sum } else { 0.0 };
            }

            let mut z2 = vec![0.0f32; h.len()];
            for o in 0..h.len() {
                let mut sum = blk.b2[o];
                for i in 0..h.len() {
                    sum += z1[i] * blk.w2.get(i, o);
                }
                z2[o] = sum + h[o];
            }

            for i in 0..h.len() {
                h[i] = if z2[i] > 0.0 { z2[i] } else { 0.0 };
            }
        }

        // Final linear layer.
        let feat = h;
        let mut logits = vec![0.0f32; self.fc.cols];
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

/// Build a small ResNet-like architecture as a [`Model`] graph.
/// The graph contains an input projection, a configurable number of
/// residual blocks and a final linear layer.
pub fn resnet_model(num_blocks: usize) -> Model {
    let mut m = Model::new();
    let input = m.add("input");
    let proj = m.add("input_proj");
    m.connect(input, proj);
    let mut prev = proj;
    for i in 0..num_blocks {
        let l1 = m.add(format!("block{}_lin1", i));
        let l2 = m.add(format!("block{}_lin2", i));
        m.connect(prev, l1);
        m.connect(l1, l2);
        // residual connection
        m.connect(prev, l2);
        prev = l2;
    }
    let fc = m.add("fc");
    m.connect(prev, fc);
    m
}

