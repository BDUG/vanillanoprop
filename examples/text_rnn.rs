use vanillanoprop::math::{self, Matrix};
use vanillanoprop::models::RNN;

fn main() {
    let data = vec![
        (vec![vec![1.0], vec![2.0], vec![3.0]], 1u8),
        (vec![vec![3.0], vec![2.0], vec![1.0]], 0u8),
        (vec![vec![0.5], vec![1.5], vec![0.5]], 1u8),
        (vec![vec![1.5], vec![0.5], vec![1.5]], 0u8),
    ];
    let mut rnn = RNN::new_lstm(1, 4, 2);
    for (i, (seq, label)) in data.iter().enumerate() {
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
        println!("sample {i} loss {loss}");
    }
}
