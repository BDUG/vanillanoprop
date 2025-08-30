use vanillanoprop::layers::LinearT;
use vanillanoprop::math::{self, Matrix};
use vanillanoprop::models::TransformerEncoder;

fn to_one_hot(seq: &[usize], vocab: usize) -> Matrix {
    let mut m = Matrix::zeros(seq.len(), vocab);
    for (t, &idx) in seq.iter().enumerate() {
        m.set(t, idx, 1.0);
    }
    m
}

fn main() {
    let vocab = 4;
    let data = vec![
        (vec![1, 2, 3], 1u8),
        (vec![3, 2, 1], 0u8),
        (vec![0, 1, 2], 1u8),
        (vec![2, 1, 0], 0u8),
    ];
    let mut enc = TransformerEncoder::new(2, vocab, 8, 2, 16, 0.1);
    let mut clf = LinearT::new(8, 2);

    for (i, (seq, label)) in data.iter().enumerate() {
        let x = to_one_hot(seq, vocab);
        let h = enc.forward_train(&x, None);
        // take representation of first token
        let mut cls = Matrix::zeros(1, h.cols);
        for c in 0..h.cols { cls.set(0, c, h.get(0, c)); }
        let logits = clf.forward_train(&cls);
        let (loss, grad, _) = math::softmax_cross_entropy(&logits, &[*label as usize], 0);
        clf.zero_grad();
        enc.zero_grad();
        let grad_cls = clf.backward(&grad);
        let mut grad_enc = Matrix::zeros(h.rows, h.cols);
        for c in 0..h.cols { grad_enc.set(0, c, grad_cls.get(0, c)); }
        enc.backward(&grad_enc);
        clf.adam_step(0.05, 0.9, 0.999, 1e-8, 0.0);
        enc.adam_step(0.05, 0.9, 0.999, 1e-8, 0.0);
        println!("sample {i} loss {loss}");
    }
}
