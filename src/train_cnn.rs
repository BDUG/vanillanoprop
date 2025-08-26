use indicatif::ProgressBar;

use crate::data::load_batches;
use crate::math::{self, Matrix};
use crate::memory;
use crate::metrics::f1_score;
use crate::models::SimpleCNN;
use crate::optim::PaperAlgo;
use crate::weights::save_cnn;

/// Train a [`SimpleCNN`] on the MNIST data using a basic SGD loop.
///
/// `opt` selects the optimisation algorithm.  `moe` and `num_experts` are
/// accepted for API compatibility but currently unused.
pub fn run(opt: &str, _moe: bool, _num_experts: usize) {
    let batches = load_batches(4);
    let mut cnn = SimpleCNN::new(10);

    let lr = 0.01f32;
    let mut paper = PaperAlgo::new(lr, 0.0);
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

                let logits_m = Matrix::from_vec(1, logits.len(), logits);
                let (loss, grad_m, preds) =
                    math::softmax_cross_entropy(&logits_m, &[*tgt as usize], 0);
                batch_loss += loss;
                let grad_logits = grad_m.data;

                // Update weights
                let (fc, bias) = cnn.parameters_mut();
                if opt == "paper" {
                    paper.update(fc, bias, &grad_logits, &feat);
                } else {
                    let rows = fc.rows;
                    let cols = fc.cols;
                    for c in 0..cols {
                        let g = grad_logits[c];
                        bias[c] -= lr * g; // mul + add
                        for r in 0..rows {
                            let val = fc.get(r, c) - lr * g * feat[r]; // 2 mul + add
                            fc.set(r, c, val);
                        }
                    }
                    let ops = cols * (2 + rows * 3); // bias update + weight update
                    math::inc_ops_by(ops);
                }

                batch_f1 += f1_score(&preds, &[*tgt as usize]);
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
            if let Err(e) = save_cnn("checkpoint_cnn.json", &cnn) {
                eprintln!("Failed to save checkpoint: {e}");
            }
        }
    }

    pb.finish_with_message("training done");

    println!("Total matrix ops: {}", math::matrix_ops_count());
    let peak = memory::peak_memory_bytes();
    println!(
        "Max memory usage: {:.2} MB",
        peak as f64 / (1024.0 * 1024.0)
    );
    if let Err(e) = save_cnn("cnn.json", &cnn) {
        eprintln!("Failed to save model: {e}");
    }
}
