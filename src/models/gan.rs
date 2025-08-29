use crate::layers::conv::Conv2d;
use crate::layers::layer::Layer;
use crate::layers::linear::LinearT;
use crate::layers::{leaky_relu, relu, sigmoid, tanh};
use crate::math::Matrix;

pub struct Generator {
    pub fc1: LinearT,
    pub fc2: LinearT,
    mask: Vec<f32>,
    out_cache: Matrix,
}

impl Generator {
    pub fn new(latent_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            fc1: LinearT::new(latent_dim, hidden_dim),
            fc2: LinearT::new(hidden_dim, output_dim),
            mask: Vec::new(),
            out_cache: Matrix::zeros(0, 0),
        }
    }

    pub fn forward_train(&mut self, z: &Matrix) -> Matrix {
        let mut h = self.fc1.forward_train(z);
        self.mask = relu::forward_matrix(&mut h);
        let mut out = self.fc2.forward_train(&h);
        tanh::forward_matrix(&mut out);
        self.out_cache = out.clone();
        out
    }

    pub fn backward(&mut self, grad_out: &Matrix) {
        let mut g = grad_out.clone();
        tanh::backward(&mut g, &self.out_cache);
        let grad_h = self.fc2.backward(&g);
        let mut grad_h_act = grad_h.clone();
        relu::backward(&mut grad_h_act, &self.mask);
        self.fc1.backward(&grad_h_act);
    }

    pub fn zero_grad(&mut self) {
        self.fc1.zero_grad();
        self.fc2.zero_grad();
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        vec![&mut self.fc1, &mut self.fc2]
    }
}

pub struct Discriminator {
    pub conv: Conv2d,
    pub fc: LinearT,
    mask: Vec<f32>,
}

impl Discriminator {
    pub fn new() -> Self {
        Self {
            conv: Conv2d::new(1, 4, 3, 2, 1),
            fc: LinearT::new(4 * 14 * 14, 1),
            mask: Vec::new(),
        }
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        let mut h = self.conv.forward_train(x);
        self.mask = leaky_relu::forward_matrix(&mut h);
        self.fc.forward_train(&h)
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let grad_fc = self.fc.backward(grad_out);
        let mut grad_h = grad_fc.clone();
        leaky_relu::backward(&mut grad_h, &self.mask);
        self.conv.backward(&grad_h)
    }

    pub fn zero_grad(&mut self) {
        self.conv.zero_grad();
        self.fc.zero_grad();
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let mut params = self.conv.parameters();
        params.push(&mut self.fc);
        params
    }
}

pub struct GAN {
    pub generator: Generator,
    pub discriminator: Discriminator,
}

impl GAN {
    pub fn new(latent_dim: usize) -> Self {
        Self {
            generator: Generator::new(latent_dim, 128, 28 * 28),
            discriminator: Discriminator::new(),
        }
    }

    pub fn train_step(&mut self, real: &Matrix, noise: &Matrix, lr: f32) -> (f32, f32) {
        // Train discriminator
        self.discriminator.zero_grad();
        let fake = self.generator.forward_train(noise);
        let real_logits = self.discriminator.forward_train(real);
        let fake_logits = self.discriminator.forward_train(&fake);
        let mut real_prob = real_logits.clone();
        sigmoid::forward_matrix(&mut real_prob);
        let mut fake_prob = fake_logits.clone();
        sigmoid::forward_matrix(&mut fake_prob);
        let mut grad_real = Matrix::zeros(real_logits.rows, real_logits.cols);
        let mut grad_fake = Matrix::zeros(fake_logits.rows, fake_logits.cols);
        let mut d_loss = 0.0f32;
        for i in 0..real_prob.data.len() {
            let p = real_prob.data[i];
            d_loss += -(p + 1e-9).ln();
            grad_real.data[i] = p - 1.0;
        }
        for i in 0..fake_prob.data.len() {
            let p = fake_prob.data[i];
            d_loss += -((1.0 - p) + 1e-9).ln();
            grad_fake.data[i] = p;
        }
        d_loss /= (real_prob.data.len() + fake_prob.data.len()) as f32;
        self.discriminator.backward(&grad_real);
        self.discriminator.backward(&grad_fake);
        for p in self.discriminator.parameters() {
            p.sgd_step(lr, 0.0);
        }

        // Train generator
        self.generator.zero_grad();
        self.discriminator.zero_grad();
        let fake = self.generator.forward_train(noise);
        let fake_logits = self.discriminator.forward_train(&fake);
        let mut fake_prob = fake_logits.clone();
        sigmoid::forward_matrix(&mut fake_prob);
        let mut grad = Matrix::zeros(fake_logits.rows, fake_logits.cols);
        let mut g_loss = 0.0f32;
        for i in 0..fake_prob.data.len() {
            let p = fake_prob.data[i];
            g_loss += -(p + 1e-9).ln();
            grad.data[i] = p - 1.0;
        }
        let grad_input = self.discriminator.backward(&grad);
        self.generator.backward(&grad_input);
        for p in self.generator.parameters() {
            p.sgd_step(lr, 0.0);
        }
        g_loss /= fake_prob.data.len() as f32;
        (d_loss, g_loss)
    }
}
