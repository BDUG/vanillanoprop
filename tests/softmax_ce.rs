use vanillanoprop::math::{self, Matrix};

fn manual_softmax_ce(logits: &[f32], target: usize) -> (f32, Vec<f32>, usize) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0f32;
    let mut probs = vec![0f32; logits.len()];
    for (i, &v) in logits.iter().enumerate() {
        let e = (v - max).exp();
        probs[i] = e;
        exp_sum += e;
    }
    for p in &mut probs {
        *p /= exp_sum;
    }
    let loss = -probs[target].ln();
    let mut grad = probs.clone();
    grad[target] -= 1.0;
    let mut pred = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &p) in probs.iter().enumerate() {
        if p > best_val {
            best_val = p;
            pred = i;
        }
    }
    (loss, grad, pred)
}

#[test]
fn softmax_ce_matches_manual() {
    let logits_vec = vec![1.0, 2.0, 0.5];
    let logits = Matrix::from_vec(1, 3, logits_vec.clone());
    let target = 1usize;

    let (loss, grad, pred) = manual_softmax_ce(&logits_vec, target);
    let (loss2, grad_m, preds) = math::softmax_cross_entropy(&logits, &[target], 0);

    assert!((loss - loss2).abs() < 1e-6);
    for (a, b) in grad.iter().zip(grad_m.data.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
    assert_eq!(preds[0], pred);
}

#[test]
fn argmax_matches_manual() {
    let v = vec![0.1, 0.9, 0.2];
    assert_eq!(math::argmax(&v), 1);
}
