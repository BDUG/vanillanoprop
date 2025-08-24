use indicatif::ProgressBar;

use crate::data::load_batches;
use crate::math;
use crate::metrics::f1_score;
use crate::models::SimpleCNN;
use crate::weights::save_cnn;

/// Train a [`SimpleCNN`] on the MNIST data using a basic SGD loop.
///
/// `opt` is kept for parity with other training binaries but currently only
/// SGD is implemented.
pub fn run(opt: &str) {
    let _ = opt; // optimizer placeholder

    let batches = load_batches(4);
    let mut cnn = SimpleCNN::new(10);

    let lr = 0.01f32;
    let epochs = 5;

    math::reset_matrix_ops();
    let pb = ProgressBar::new(epochs as u64);
    let mut best_f1 = f32::NEG_INFINITY;

    for epoch in 0..epochs {
        let mut last_loss = 0.0f32;
        let mut f1_sum = 0.0f32;
        let mut sample_cnt = 0.0f32;

        for batch in &batches {
            let mut batch_loss = 0.0f32;
            let mut batch_f1 = 0.0f32;

            for (img, tgt) in batch {
                let (feat, logits) = cnn.forward(img);

                // Softmax
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

                let loss = -probs[*tgt as usize].ln();
                batch_loss += loss;

                // Gradient of cross-entropy w.r.t logits
                let mut grad_logits = probs.clone();
                grad_logits[*tgt as usize] -= 1.0;

                // Update weights
                let (fc, bias) = cnn.parameters_mut();
                let rows = fc.rows;
                let cols = fc.cols;
                for c in 0..cols {
                    let g = grad_logits[c];
                    bias[c] -= lr * g;
                    for r in 0..rows {
                        let val = fc.get(r, c) - lr * g * feat[r];
                        fc.set(r, c, val);
                    }
                }
                math::inc_ops();

                // Prediction for metrics
                let mut best = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for (i, &v) in probs.iter().enumerate() {
                    if v > best_val {
                        best_val = v;
                        best = i;
                    }
                }
                batch_f1 += f1_score(&[best], &[*tgt as usize]);
            }

            let bsz = batch.len() as f32;
            batch_loss /= bsz;
            let batch_f1_avg = batch_f1 / bsz;
            last_loss = batch_loss;
            f1_sum += batch_f1;
            sample_cnt += bsz;
            println!("loss {batch_loss:.4} f1 {batch_f1_avg:.4}");
        }

        let avg_f1 = f1_sum / if sample_cnt > 0.0 { sample_cnt } else { 1.0 };
        pb.set_message(format!("epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}"));
        pb.inc(1);

        if avg_f1 > best_f1 {
            println!("Checkpoint saved at epoch {epoch}: avg F1 improved to {avg_f1:.4}");
            best_f1 = avg_f1;
            save_cnn("checkpoint_cnn.json", &cnn);
        }
    }

    pb.finish_with_message("training done");

    println!("Total matrix ops: {}", math::matrix_ops_count());
    save_cnn("cnn.json", &cnn);
}

