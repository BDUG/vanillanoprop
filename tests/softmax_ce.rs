use rand::Rng;
use vanillanoprop::math::{self, Matrix};

// Baseline implementation of softmax cross entropy used to verify the
// refactored version. This mirrors the previous logic from the library.
fn baseline_softmax_cross_entropy(
    logits: &Matrix,
    targets: &[usize],
    row_offset: usize,
) -> (f32, Matrix, Vec<usize>) {
    let probs = logits.softmax();
    let mut grad = probs.clone();
    let mut loss = 0.0f32;
    let mut preds = Vec::new();
    let mut cnt = 0.0f32;

    for (i, &tok) in targets.iter().enumerate() {
        let row = i + row_offset;
        if row >= logits.rows {
            break;
        }

        let mut best_tok = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for t in 0..logits.cols {
            let p = probs.get(row, t);
            if p > best_val {
                best_val = p;
                best_tok = t;
            }
        }

        let p = probs.get(row, tok);
        loss += -(p + 1e-9).ln();
        grad.set(row, tok, grad.get(row, tok) - 1.0);
        preds.push(best_tok);
        cnt += 1.0;
    }

    if cnt > 0.0 {
        loss /= cnt;
        for v in grad.data.iter_mut() {
            *v /= cnt;
        }
    }

    (loss, grad, preds)
}

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

#[test]
fn refactored_matches_baseline_with_extreme_logits() {
    let rows = 4;
    let cols = 6;
    let mut rng = rand::thread_rng();
    let logits_vec: Vec<f32> = (0..rows * cols)
        .map(|_| rng.gen_range(-1000.0..1000.0))
        .collect();
    let logits = Matrix::from_vec(rows, cols, logits_vec);
    let targets: Vec<usize> = (0..rows).map(|_| rng.gen_range(0..cols)).collect();

    let (loss_old, grad_old, preds_old) = baseline_softmax_cross_entropy(&logits, &targets, 0);
    let (loss_new, grad_new, preds_new) = math::softmax_cross_entropy(&logits, &targets, 0);

    assert!((loss_old - loss_new).abs() < 1e-6);
    for (a, b) in grad_old.data.iter().zip(grad_new.data.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
    assert_eq!(preds_old, preds_new);
}
