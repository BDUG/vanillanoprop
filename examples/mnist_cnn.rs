use vanillanoprop::data::{Dataset, Mnist};
use vanillanoprop::layers::{Layer, LinearT, MixtureOfExpertsT};
use vanillanoprop::math::{self, Matrix};
use vanillanoprop::models::SimpleCNN;
use vanillanoprop::weights::save_moe;

fn main() {
    // Load MNIST and group into mini-batches using the Dataset API.
    let batches = Mnist::batch(32);
    let cnn = SimpleCNN::new(10);
    let lr = 0.01f32;
    let experts: Vec<Box<dyn Layer>> = (0..3)
        .map(|_| Box::new(LinearT::new(28 * 28, 10)) as Box<dyn Layer>)
        .collect();
    let mut moe = MixtureOfExpertsT::new(28 * 28, experts, 1);

    for (i, batch) in batches.iter().take(5).enumerate() {
        let mut loss_sum = 0.0f32;
        for (img, label) in batch {
            let (feat, _logits) = cnn.forward(img);
            let feat_m = Matrix::from_vec(1, feat.len(), feat.clone());
            let logits_m = moe.forward_train(&feat_m);
            let (loss, grad, _) = math::softmax_cross_entropy(&logits_m, &[*label as usize], 0);
            loss_sum += loss;

            moe.zero_grad();
            moe.backward(&grad);
            for p in moe.parameters() {
                p.sgd_step(lr, 0.0);
            }
        }
        println!("batch {i} loss {}", loss_sum / batch.len() as f32);
    }

    if let Err(e) = save_moe("moe.json", &mut moe) {
        eprintln!("failed to save MoE weights: {e}");
    }
}
