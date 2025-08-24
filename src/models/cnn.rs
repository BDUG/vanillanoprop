use crate::math::Matrix;
use rand::Rng;

/// A very small convolutional network used for demonstration purposes.
///
/// The network applies a single 3x3 convolution with ReLU activation
/// followed by a linear layer that maps the flattened feature map to the
/// desired number of classes.
pub struct SimpleCNN {
    kernel: [[f32; 3]; 3],
    fc: Matrix,
    bias: Vec<f32>,
}

impl SimpleCNN {
    /// Create a new CNN with random weights for the fully connected layer.
    /// The convolution kernel is initialised with a simple mean blur.
    pub fn new(num_classes: usize) -> Self {
        // 3x3 mean kernel
        let kernel = [[1.0 / 9.0; 3]; 3];
        let mut rng = rand::thread_rng();
        let mut w = Vec::with_capacity(28 * 28 * num_classes);
        for _ in 0..(28 * 28 * num_classes) {
            w.push(rng.gen_range(-0.01..0.01));
        }
        let fc = Matrix::from_vec(28 * 28, num_classes, w);
        let bias = vec![0.0; num_classes];
        Self { kernel, fc, bias }
    }

    fn convolve(&self, img: &[u8]) -> Vec<f32> {
        let width = 28;
        let height = 28;
        let mut out = vec![0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let mut acc = 0.0f32;
                for ky in 0..3 {
                    for kx in 0..3 {
                        let ix = x as isize + kx as isize - 1;
                        let iy = y as isize + ky as isize - 1;
                        if ix >= 0 && ix < width as isize && iy >= 0 && iy < height as isize {
                            let idx = iy as usize * width + ix as usize;
                            acc += img[idx] as f32 * self.kernel[ky][kx];
                        }
                    }
                }
                if acc < 0.0 {
                    acc = 0.0; // ReLU
                }
                out[y * width + x] = acc;
            }
        }
        out
    }

    /// Forward pass returning the convolution features and logits.
    pub fn forward(&self, img: &[u8]) -> (Vec<f32>, Vec<f32>) {
        let feat = self.convolve(img); // 28x28 -> 784 features
        let rows = self.fc.rows;
        let cols = self.fc.cols;
        let mut logits = vec![0f32; cols];
        for c in 0..cols {
            let mut sum = self.bias[c];
            for r in 0..rows {
                sum += feat[r] * self.fc.get(r, c);
            }
            logits[c] = sum;
        }
        (feat, logits)
    }

    /// Predict the class for a single image.
    pub fn predict(&self, img: &[u8]) -> usize {
        let (_feat, logits) = self.forward(img);
        // Argmax over logits
        let mut best = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        best
    }

    /// Access immutable parameters.
    pub fn parameters(&self) -> (&Matrix, &Vec<f32>) {
        (&self.fc, &self.bias)
    }

    /// Access mutable parameters.
    pub fn parameters_mut(&mut self) -> (&mut Matrix, &mut Vec<f32>) {
        (&mut self.fc, &mut self.bias)
    }
}

