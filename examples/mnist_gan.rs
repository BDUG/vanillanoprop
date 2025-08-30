use rand::Rng;
use vanillanoprop::data::{DataLoader, Mnist};
use vanillanoprop::math::Matrix;
use vanillanoprop::models::GAN;
use vanillanoprop::rng::rng_from_env;

fn main() {
    let mut gan = GAN::new(100);
    let mut rng = rng_from_env();
    for (i, batch) in DataLoader::<Mnist>::new(32, true, None).take(2).enumerate() {
        let mut d_loss_sum = 0.0f32;
        let mut g_loss_sum = 0.0f32;
        for (img, _) in batch {
            let real: Vec<f32> = img.iter().map(|&p| p as f32 / 255.0).collect();
            let real_m = Matrix::from_vec(1, real.len(), real);
            let noise: Vec<f32> = (0..100).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
            let noise_m = Matrix::from_vec(1, 100, noise);
            let (d_loss, g_loss) = gan.train_step(&real_m, &noise_m, 0.0002);
            d_loss_sum += d_loss;
            g_loss_sum += g_loss;
        }
        println!(
            "batch {i} d_loss {} g_loss {}",
            d_loss_sum / batch.len() as f32,
            g_loss_sum / batch.len() as f32
        );
    }
}
