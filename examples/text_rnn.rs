use vanillanoprop::data::Dataset;
use vanillanoprop::math::{self, Matrix};
use vanillanoprop::models::RNN;

// Tiny in-memory dataset of sequences with binary labels.
struct ToyText;

impl Dataset for ToyText {
    type Item = (Vec<Vec<f32>>, u8);

    fn load() -> Vec<Self::Item> {
        vec![
            (vec![vec![1.0], vec![2.0], vec![3.0]], 1),
            (vec![vec![3.0], vec![2.0], vec![1.0]], 0),
            (vec![vec![0.5], vec![1.5], vec![0.5]], 1),
            (vec![vec![1.5], vec![0.5], vec![1.5]], 0),
        ]
    }
}

fn main() {
    let batches = ToyText::batch(2);
    let mut rnn = RNN::new_lstm(1, 4, 2);

    for (i, batch) in batches.iter().enumerate() {
        for (seq, label) in batch {
            // Convert sequence to Matrix (time_steps x input_dim)
            let mut mat = Matrix::zeros(seq.len(), 1);
            for (t, token) in seq.iter().enumerate() {
                mat.set(t, 0, token[0]);
            }
            let logits = rnn.forward_train(&mat);
            let (loss, grad, _) = math::softmax_cross_entropy(&logits, &[*label as usize], 0);
            rnn.zero_grad();
            rnn.backward(&grad);
            rnn.adam_step(0.05, 0.9, 0.999, 1e-8, 0.0);
            println!("batch {i} sample loss {loss}");
        }
    }
}
