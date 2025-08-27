use vanillanoprop::layers::{MixtureOfExpertsT, FeedForwardT, Activation, Layer};
use vanillanoprop::math::Matrix;
use vanillanoprop::tensor::Tensor;

fn main() {
    // Build three feed-forward expert networks
    let experts: Vec<Box<dyn Layer>> = (0..3)
        .map(|_| Box::new(FeedForwardT::new(4, 8, Activation::ReLU)) as Box<dyn Layer>)
        .collect();

    // Create mixture of experts with top-1 gating (sparse routing)
    let mut moe = MixtureOfExpertsT::new(4, experts, 1);

    // Two sample inputs
    let input = Matrix::from_vec(2, 4, vec![1.0, 2.0, 3.0, 4.0,
                                           4.0, 3.0, 2.0, 1.0]);

    // Forward pass through the mixture
    let output = moe.forward_local(&input);
    println!("Output: {:?}", output.data);

    // Compute gating probabilities separately for illustration
    let mut logits = moe.gate.forward_local(&input);
    if moe.top_k < logits.cols {
        for r in 0..logits.rows {
            let row_start = r * logits.cols;
            let mut indices: Vec<usize> = (0..logits.cols).collect();
            indices.select_nth_unstable_by(moe.top_k, |&a, &b| {
                logits.data[row_start + b]
                    .partial_cmp(&logits.data[row_start + a])
                    .unwrap()
            });
            for &idx in indices[moe.top_k..].iter() {
                logits.data[row_start + idx] = f32::NEG_INFINITY;
            }
        }
    }
    let probs = moe.softmax.forward(&Tensor::from_matrix(logits));
    println!("Gating probabilities: {:?}", probs.data.data);
}
