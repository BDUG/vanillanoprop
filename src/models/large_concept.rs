use crate::math::{self, Matrix};
use crate::model::Model;
use crate::rng::rng_from_env;
use rand::Rng;

/// A simple multi-layer perceptron used as a placeholder for a more complex
/// "Large Concept" model.
///
/// The network maps a flattened 28x28 image through two hidden
/// representations before producing class logits. ReLU is used as the hidden
/// activation and a small L2 regularisation term is applied during training.
pub struct LargeConceptModel {
    pub w1: Matrix,
    pub b1: Vec<f32>,
    pub w2: Matrix,
    pub b2: Vec<f32>,
    pub w3: Matrix,
    pub b3: Vec<f32>,
}

impl LargeConceptModel {
    /// Create a new model with random weights.
    pub fn new(
        input_dim: usize,
        hidden_dim1: usize,
        hidden_dim2: usize,
        num_classes: usize,
    ) -> Self {
        let mut rng = rng_from_env();
        let mut w1 = Vec::with_capacity(input_dim * hidden_dim1);
        for _ in 0..input_dim * hidden_dim1 {
            w1.push(rng.gen_range(-0.01..0.01));
        }
        let mut w2 = Vec::with_capacity(hidden_dim1 * hidden_dim2);
        for _ in 0..hidden_dim1 * hidden_dim2 {
            w2.push(rng.gen_range(-0.01..0.01));
        }
        let mut w3 = Vec::with_capacity(hidden_dim2 * num_classes);
        for _ in 0..hidden_dim2 * num_classes {
            w3.push(rng.gen_range(-0.01..0.01));
        }
        Self {
            w1: Matrix::from_vec(input_dim, hidden_dim1, w1),
            b1: vec![0.0; hidden_dim1],
            w2: Matrix::from_vec(hidden_dim1, hidden_dim2, w2),
            b2: vec![0.0; hidden_dim2],
            w3: Matrix::from_vec(hidden_dim2, num_classes, w3),
            b3: vec![0.0; num_classes],
        }
    }

    /// Forward pass returning the final hidden features and logits.
    pub fn forward(&self, img: &[u8]) -> (Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = img.iter().map(|&p| p as f32 / 255.0).collect();
        let mut h1 = vec![0f32; self.w1.cols];
        for c in 0..self.w1.cols {
            let mut sum = self.b1[c];
            for r in 0..self.w1.rows {
                sum += x[r] * self.w1.get(r, c);
            }
            h1[c] = if sum > 0.0 { sum } else { 0.0 };
        }
        let mut h2 = vec![0f32; self.w2.cols];
        for c in 0..self.w2.cols {
            let mut sum = self.b2[c];
            for r in 0..self.w2.rows {
                sum += h1[r] * self.w2.get(r, c);
            }
            h2[c] = if sum > 0.0 { sum } else { 0.0 };
        }
        let mut logits = vec![0f32; self.w3.cols];
        for c in 0..self.w3.cols {
            let mut sum = self.b3[c];
            for r in 0..self.w3.rows {
                sum += h2[r] * self.w3.get(r, c);
            }
            logits[c] = sum;
        }
        (h2, logits)
    }

    /// Predict the class for a single image.
    pub fn predict(&self, img: &[u8]) -> usize {
        let (_h, logits) = self.forward(img);
        math::argmax(&logits)
    }

    /// Perform a single training step using SGD and return the loss and
    /// predicted class.
    pub fn train_step(
        &mut self,
        img: &[u8],
        target: usize,
        lr: f32,
        l2: f32,
    ) -> (f32, usize) {
        let x: Vec<f32> = img.iter().map(|&p| p as f32 / 255.0).collect();

        // forward pass keeping intermediate activations
        let mut h1 = vec![0f32; self.w1.cols];
        for c in 0..self.w1.cols {
            let mut sum = self.b1[c];
            for r in 0..self.w1.rows {
                sum += x[r] * self.w1.get(r, c);
            }
            h1[c] = if sum > 0.0 { sum } else { 0.0 };
        }
        let mut h2 = vec![0f32; self.w2.cols];
        for c in 0..self.w2.cols {
            let mut sum = self.b2[c];
            for r in 0..self.w2.rows {
                sum += h1[r] * self.w2.get(r, c);
            }
            h2[c] = if sum > 0.0 { sum } else { 0.0 };
        }
        let mut logits = vec![0f32; self.w3.cols];
        for c in 0..self.w3.cols {
            let mut sum = self.b3[c];
            for r in 0..self.w3.rows {
                sum += h2[r] * self.w3.get(r, c);
            }
            logits[c] = sum;
        }

        let logits_m = Matrix::from_vec(1, self.w3.cols, logits.clone());
        let (mut loss, grad_m, preds) =
            math::softmax_cross_entropy(&logits_m, &[target], 0);
        let grad_logits = grad_m.data; // length num_classes

        // L2 regularisation
        let l2_term = self.w1.data.iter().chain(self.w2.data.iter()).chain(self.w3.data.iter()).fold(0.0, |acc, w| acc + w * w);
        loss += 0.5 * l2 * l2_term;

        // gradient wrt second hidden layer
        let mut grad_h2 = vec![0f32; self.w3.rows];
        for r in 0..self.w3.rows {
            let mut sum = 0.0f32;
            for c in 0..self.w3.cols {
                sum += self.w3.get(r, c) * grad_logits[c];
            }
            if h2[r] <= 0.0 {
                sum = 0.0;
            }
            grad_h2[r] = sum;
        }

        // gradient wrt first hidden layer
        let mut grad_h1 = vec![0f32; self.w2.rows];
        for r in 0..self.w2.rows {
            let mut sum = 0.0f32;
            for c in 0..self.w2.cols {
                sum += self.w2.get(r, c) * grad_h2[c];
            }
            if h1[r] <= 0.0 {
                sum = 0.0;
            }
            grad_h1[r] = sum;
        }

        // update third layer weights and bias
        for c in 0..self.w3.cols {
            for r in 0..self.w3.rows {
                let idx = r * self.w3.cols + c;
                let grad = h2[r] * grad_logits[c] + l2 * self.w3.data[idx];
                self.w3.data[idx] -= lr * grad;
            }
            self.b3[c] -= lr * grad_logits[c];
        }

        // update second layer weights and bias
        for c in 0..self.w2.cols {
            for r in 0..self.w2.rows {
                let idx = r * self.w2.cols + c;
                let grad = h1[r] * grad_h2[c] + l2 * self.w2.data[idx];
                self.w2.data[idx] -= lr * grad;
            }
            self.b2[c] -= lr * grad_h2[c];
        }

        // update first layer weights and bias
        for c in 0..self.w1.cols {
            for r in 0..self.w1.rows {
                let idx = r * self.w1.cols + c;
                let grad = x[r] * grad_h1[c] + l2 * self.w1.data[idx];
                self.w1.data[idx] -= lr * grad;
            }
            self.b1[c] -= lr * grad_h1[c];
        }

        (loss, preds[0])
    }

    /// Batched training step operating on a collection of images and targets.
    pub fn train_batch(
        &mut self,
        imgs: &[Vec<u8>],
        targets: &[usize],
        lr: f32,
        l2: f32,
    ) -> (f32, Vec<usize>) {
        let mut loss = 0.0f32;
        let mut preds = Vec::with_capacity(imgs.len());
        for (img, &t) in imgs.iter().zip(targets.iter()) {
            let (l, p) = self.train_step(img, t, lr, l2);
            loss += l;
            preds.push(p);
        }
        if !imgs.is_empty() {
            loss /= imgs.len() as f32;
        }
        (loss, preds)
    }

    /// Access immutable parameters for serialisation.
    pub fn parameters(
        &self,
    ) -> (&Matrix, &Vec<f32>, &Matrix, &Vec<f32>, &Matrix, &Vec<f32>) {
        (
            &self.w1,
            &self.b1,
            &self.w2,
            &self.b2,
            &self.w3,
            &self.b3,
        )
    }

    /// Access mutable parameters for deserialisation.
    pub fn parameters_mut(
        &mut self,
    ) -> (
        &mut Matrix,
        &mut Vec<f32>,
        &mut Matrix,
        &mut Vec<f32>,
        &mut Matrix,
        &mut Vec<f32>,
    ) {
        (
            &mut self.w1,
            &mut self.b1,
            &mut self.w2,
            &mut self.b2,
            &mut self.w3,
            &mut self.b3,
        )
    }
}

/// Build the "Large Concept" network as a [`Model`] graph with an input
/// layer followed by two hidden ReLU layers and a final linear classification
/// layer.
pub fn large_concept_model(
    input_dim: usize,
    hidden_dim1: usize,
    hidden_dim2: usize,
    num_classes: usize,
) -> Model {
    let mut m = Model::new();
    let input = m.add(format!("input{}", input_dim));
    let fc1 = m.add(format!("linear{}", hidden_dim1));
    let relu1 = m.add("relu");
    let fc2 = m.add(format!("linear{}", hidden_dim2));
    let relu2 = m.add("relu");
    let fc3 = m.add(format!("linear{}", num_classes));
    m.connect(input, fc1);
    m.connect(fc1, relu1);
    m.connect(relu1, fc2);
    m.connect(fc2, relu2);
    m.connect(relu2, fc3);
    m
}
