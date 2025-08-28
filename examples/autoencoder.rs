use vanillanoprop::data::{Dataset, Mnist};
use vanillanoprop::math::Matrix;
use vanillanoprop::models::VAE;

fn main() {
    // Use dataset utilities to load MNIST in mini-batches
    let batches = Mnist::batch(16);
    let mut vae = VAE::new(28 * 28, 128, 32);

    for (i, batch) in batches.iter().take(3).enumerate() {
        let mut loss_sum = 0.0f32;
        for (img, _) in batch {
            let input: Vec<f32> = img.iter().map(|&p| p as f32 / 255.0).collect();
            let x = Matrix::from_vec(1, input.len(), input);
            let (recon, mu, logvar) = vae.forward_train(&x);

            // reconstruction loss (MSE)
            let mut recon_grad = Matrix::zeros(recon.rows, recon.cols);
            let mut recon_loss = 0.0f32;
            for j in 0..recon.data.len() {
                let diff = recon.data[j] - x.data[j];
                recon_loss += diff * diff;
                recon_grad.data[j] = 2.0 * diff / recon.data.len() as f32;
            }

            // KL divergence gradients
            let mut grad_mu = Matrix::zeros(mu.rows, mu.cols);
            let mut grad_logvar = Matrix::zeros(logvar.rows, logvar.cols);
            let mut kl = 0.0f32;
            for j in 0..mu.data.len() {
                let m = mu.data[j];
                let lv = logvar.data[j];
                kl += -0.5 * (1.0 + lv - m * m - lv.exp());
                grad_mu.data[j] = m;
                grad_logvar.data[j] = 0.5 * (lv.exp() - 1.0);
            }

            vae.zero_grad();
            vae.backward(&recon_grad, &grad_mu, &grad_logvar);
            for p in vae.parameters() {
                p.adam_step(0.001, 0.9, 0.999, 1e-8, 0.0);
            }

            loss_sum += recon_loss / recon.data.len() as f32 + kl / mu.data.len() as f32;
        }
        println!("batch {i} loss {}", loss_sum / batch.len() as f32);
    }
}
