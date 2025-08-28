use crate::layers::linear::LinearT;
use crate::layers::{relu, sigmoid};
use crate::math::Matrix;
use crate::model::Model;
use crate::rng::rng_from_env;
use rand_distr::{Distribution, StandardNormal};

pub struct VAE {
    pub enc_fc1: LinearT,
    pub enc_mu: LinearT,
    pub enc_logvar: LinearT,
    pub dec_fc1: LinearT,
    pub dec_fc2: LinearT,
    // caches for backward
    enc_mask: Vec<f32>,
    dec_mask: Vec<f32>,
    pub mu: Matrix,
    pub logvar: Matrix,
    std: Matrix,
    eps: Matrix,
    z: Matrix,
    recon: Matrix,
}

impl VAE {
    pub fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize) -> Self {
        Self {
            enc_fc1: LinearT::new(input_dim, hidden_dim),
            enc_mu: LinearT::new(hidden_dim, latent_dim),
            enc_logvar: LinearT::new(hidden_dim, latent_dim),
            dec_fc1: LinearT::new(latent_dim, hidden_dim),
            dec_fc2: LinearT::new(hidden_dim, input_dim),
            enc_mask: Vec::new(),
            dec_mask: Vec::new(),
            mu: Matrix::zeros(0, 0),
            logvar: Matrix::zeros(0, 0),
            std: Matrix::zeros(0, 0),
            eps: Matrix::zeros(0, 0),
            z: Matrix::zeros(0, 0),
            recon: Matrix::zeros(0, 0),
        }
    }

    pub fn forward_train(&mut self, x: &Matrix) -> (Matrix, Matrix, Matrix) {
        // encoder
        let mut h1 = self.enc_fc1.forward_train(x);
        self.enc_mask = relu::forward_matrix(&mut h1);
        self.mu = self.enc_mu.forward_train(&h1);
        self.logvar = self.enc_logvar.forward_train(&h1);

        // reparameterization
        let mut rng = rng_from_env();
        let mut z = Matrix::zeros(self.mu.rows, self.mu.cols);
        let mut std = Matrix::zeros(self.mu.rows, self.mu.cols);
        let mut eps = Matrix::zeros(self.mu.rows, self.mu.cols);
        for i in 0..self.mu.data.len() {
            let lv = self.logvar.data[i];
            let s = (0.5 * lv).exp();
            let e: f32 = StandardNormal.sample(&mut rng);
            std.data[i] = s;
            eps.data[i] = e;
            z.data[i] = self.mu.data[i] + e * s;
        }
        self.std = std;
        self.eps = eps;
        self.z = z.clone();

        // decoder
        let mut h2 = self.dec_fc1.forward_train(&z);
        self.dec_mask = relu::forward_matrix(&mut h2);
        let mut recon = self.dec_fc2.forward_train(&h2);
        sigmoid::forward_matrix(&mut recon);
        self.recon = recon.clone();
        (recon, self.mu.clone(), self.logvar.clone())
    }

    pub fn decode(&mut self, z: &Matrix) -> Matrix {
        let mut h1 = self.dec_fc1.forward_local(z);
        let _ = relu::forward_matrix(&mut h1);
        let mut out = self.dec_fc2.forward_local(&h1);
        sigmoid::forward_matrix(&mut out);
        out
    }

    pub fn backward(
        &mut self,
        grad_recon: &Matrix,
        grad_mu_kl: &Matrix,
        grad_logvar_kl: &Matrix,
    ) {
        // decoder
        let mut g = grad_recon.clone();
        sigmoid::backward(&mut g, &self.recon);
        let grad_h2 = self.dec_fc2.backward(&g);
        let mut grad_h2_act = grad_h2.clone();
        for (i, v) in grad_h2_act.data.iter_mut().enumerate() {
            *v *= self.dec_mask[i];
        }
        let grad_z = self.dec_fc1.backward(&grad_h2_act);

        // gradients for mu and logvar
        let grad_mu = grad_z.add(grad_mu_kl);
        let mut grad_logvar = Matrix::zeros(self.logvar.rows, self.logvar.cols);
        for i in 0..grad_logvar.data.len() {
            let g = grad_z.data[i] * self.eps.data[i] * 0.5 * self.std.data[i];
            grad_logvar.data[i] = g + grad_logvar_kl.data[i];
        }
        let grad_h1_mu = self.enc_mu.backward(&grad_mu);
        let grad_h1_logvar = self.enc_logvar.backward(&grad_logvar);
        let mut grad_h1 = grad_h1_mu.add(&grad_h1_logvar);
        for (i, v) in grad_h1.data.iter_mut().enumerate() {
            *v *= self.enc_mask[i];
        }
        self.enc_fc1.backward(&grad_h1);
    }

    pub fn zero_grad(&mut self) {
        self.enc_fc1.zero_grad();
        self.enc_mu.zero_grad();
        self.enc_logvar.zero_grad();
        self.dec_fc1.zero_grad();
        self.dec_fc2.zero_grad();
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        vec![
            &mut self.enc_fc1,
            &mut self.enc_mu,
            &mut self.enc_logvar,
            &mut self.dec_fc1,
            &mut self.dec_fc2,
        ]
    }
}

/// Build a Variational Autoencoder architecture as a [`Model`] graph.
/// The graph captures the encoder, latent sampling and decoder
/// components of the VAE.
pub fn vae_model() -> Model {
    let mut m = Model::new();
    let input = m.add("input");
    let enc_fc1 = m.add("enc_fc1");
    let enc_mu = m.add("enc_mu");
    let enc_logvar = m.add("enc_logvar");
    let latent = m.add("latent");
    let dec_fc1 = m.add("dec_fc1");
    let dec_fc2 = m.add("dec_fc2");
    m.connect(input, enc_fc1);
    m.connect(enc_fc1, enc_mu);
    m.connect(enc_fc1, enc_logvar);
    m.connect(enc_mu, latent);
    m.connect(enc_logvar, latent);
    m.connect(latent, dec_fc1);
    m.connect(dec_fc1, dec_fc2);
    m
}
