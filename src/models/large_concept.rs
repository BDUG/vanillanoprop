use crate::math::{self, Matrix};
use crate::rng::rng_from_env;
use rand::Rng;

/// A simple two layer perceptron used as a placeholder for a more complex
/// "Large Concept" model.
///
/// The network maps a flattened 28x28 image to a hidden representation and
/// finally to class logits.  ReLU is used as the hidden activation.
pub struct LargeConceptModel {
    pub w1: Matrix,
    pub b1: Vec<f32>,
    pub w2: Matrix,
    pub b2: Vec<f32>,
}

impl LargeConceptModel {
    /// Create a new model with random weights.
    pub fn new(input_dim: usize, hidden_dim: usize, num_classes: usize) -> Self {
        let mut rng = rng_from_env();
        let mut w1 = Vec::with_capacity(input_dim * hidden_dim);
        for _ in 0..input_dim * hidden_dim {
            w1.push(rng.gen_range(-0.01..0.01));
        }
        let mut w2 = Vec::with_capacity(hidden_dim * num_classes);
        for _ in 0..hidden_dim * num_classes {
            w2.push(rng.gen_range(-0.01..0.01));
        }
        Self {
            w1: Matrix::from_vec(input_dim, hidden_dim, w1),
            b1: vec![0.0; hidden_dim],
            w2: Matrix::from_vec(hidden_dim, num_classes, w2),
            b2: vec![0.0; num_classes],
        }
    }

    /// Forward pass returning hidden features and logits.
    pub fn forward(&self, img: &[u8]) -> (Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = img.iter().map(|&p| p as f32 / 255.0).collect();
        let mut hidden = vec![0f32; self.w1.cols];
        for c in 0..self.w1.cols {
            let mut sum = self.b1[c];
            for r in 0..self.w1.rows {
                sum += x[r] * self.w1.get(r, c);
            }
            hidden[c] = if sum > 0.0 { sum } else { 0.0 };
        }
        let mut logits = vec![0f32; self.w2.cols];
        for c in 0..self.w2.cols {
            let mut sum = self.b2[c];
            for r in 0..self.w2.rows {
                sum += hidden[r] * self.w2.get(r, c);
            }
            logits[c] = sum;
        }
        (hidden, logits)
    }

    /// Predict the class for a single image.
    pub fn predict(&self, img: &[u8]) -> usize {
        let (_h, logits) = self.forward(img);
        math::argmax(&logits)
    }

    /// Perform a single training step using SGD and return the loss and
    /// predicted class.
    pub fn train_step(&mut self, img: &[u8], target: usize, lr: f32) -> (f32, usize) {
        let x: Vec<f32> = img.iter().map(|&p| p as f32 / 255.0).collect();
        let (hidden, logits) = self.forward(img);
        let logits_m = Matrix::from_vec(1, self.w2.cols, logits.clone());
        let (loss, grad_m, preds) = math::softmax_cross_entropy(&logits_m, &[target], 0);
        let grad_logits = grad_m.data; // length num_classes

        // gradient wrt hidden activations (before ReLU)
        let mut grad_hidden = vec![0f32; self.w2.rows];
        for r in 0..self.w2.rows {
            let mut sum = 0.0f32;
            for c in 0..self.w2.cols {
                sum += self.w2.get(r, c) * grad_logits[c];
            }
            if hidden[r] <= 0.0 {
                sum = 0.0;
            }
            grad_hidden[r] = sum;
        }

        // update second layer weights and bias
        for c in 0..self.w2.cols {
            for r in 0..self.w2.rows {
                let idx = r * self.w2.cols + c;
                self.w2.data[idx] -= lr * hidden[r] * grad_logits[c];
            }
            self.b2[c] -= lr * grad_logits[c];
        }

        // update first layer weights and bias
        for c in 0..self.w1.cols {
            for r in 0..self.w1.rows {
                let idx = r * self.w1.cols + c;
                self.w1.data[idx] -= lr * x[r] * grad_hidden[c];
            }
            self.b1[c] -= lr * grad_hidden[c];
        }

        (loss, preds[0])
    }

    /// Access immutable parameters for serialisation.
    pub fn parameters(&self) -> (&Matrix, &Vec<f32>, &Matrix, &Vec<f32>) {
        (&self.w1, &self.b1, &self.w2, &self.b2)
    }

    /// Access mutable parameters for deserialisation.
    pub fn parameters_mut(&mut self) -> (&mut Matrix, &mut Vec<f32>, &mut Matrix, &mut Vec<f32>) {
        (&mut self.w1, &mut self.b1, &mut self.w2, &mut self.b2)
    }
}
