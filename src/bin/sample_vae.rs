use std::env;
use vanillanoprop::math::Matrix;
use vanillanoprop::rng::rng_from_env;
use vanillanoprop::weights::load_vae;
use rand_distr::{Distribution, StandardNormal};

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| "vae.json".to_string());
    let input_dim = 28 * 28;
    let hidden_dim = 400;
    let latent_dim = 20;
    let mut vae = load_vae(&path, input_dim, hidden_dim, latent_dim).expect("load vae");
    let mut rng = rng_from_env();
    let mut z = Matrix::zeros(1, latent_dim);
    for i in 0..latent_dim {
        let e: f32 = StandardNormal.sample(&mut rng);
        z.set(0, i, e);
    }
    let sample = vae.decode(&z);
    println!("sample first values: {:?}", &sample.data[..10.min(sample.data.len())]);
}
