use vanillanoprop::data::{DataLoader, Mnist};
use vanillanoprop::math::{kl_divergence, mse_loss, Matrix};
use vanillanoprop::model::Model;
use vanillanoprop::models::VAE;
use vanillanoprop::optim::{Adam, MseLoss};
use vanillanoprop::weights::save_vae;

mod common;

fn main() {
    let _ = common::init_logging();
    let pairs: Vec<(Vec<u8>, usize)> = DataLoader::<Mnist>::new(1, true, None)
        .take(10)
        .flat_map(|b| b.iter().cloned())
        .collect();
    let input_dim = 28 * 28;
    let hidden_dim = 400;
    let latent_dim = 20;
    let mut vae = VAE::new(input_dim, hidden_dim, latent_dim);
    let mut trainer = Model::new();
    trainer.compile(Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0), MseLoss::new());

    for epoch in 0..3 {
        let mut total = 0.0f32;
        for (pixels, _) in &pairs {
            let x_vec: Vec<f32> = pixels.iter().map(|&p| p as f32 / 255.0).collect();
            let x = Matrix::from_vec(1, input_dim, x_vec);
            let (recon, mu, logvar) = vae.forward_train(&x);
            let (recon_loss, grad_recon) = mse_loss(&recon, &x);
            let (kl_loss, grad_mu_kl, grad_logvar_kl) = kl_divergence(&mu, &logvar);
            vae.zero_grad();
            vae.backward(&grad_recon, &grad_mu_kl, &grad_logvar_kl);
            let mut params = vae.parameters();
            trainer.fit(&mut params);
            total += recon_loss + kl_loss;
        }
        vanillanoprop::info!("epoch {epoch} loss {:.4}", total / pairs.len() as f32);
    }

    if let Err(e) = save_vae("vae.json", &vae) {
        vanillanoprop::error!("failed to save model: {e}");
    }
}
