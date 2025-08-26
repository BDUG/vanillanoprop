use indicatif::ProgressBar;

use crate::data::load_batches;
use crate::math::{self, Matrix};
use crate::memory;
use crate::metrics::f1_score;
use crate::models::SimpleCNN;
use crate::optim::lr_scheduler::{
    ConstantLr, CosineLr, LearningRateSchedule, LrScheduleConfig, StepLr,
};
use crate::optim::Hrm;
use crate::weights::save_cnn;

/// Train a [`SimpleCNN`] on the MNIST data using a basic SGD loop.
///
/// `opt` selects the optimisation algorithm.  `moe` and `num_experts` are
/// accepted for API compatibility but currently unused.
pub fn run(opt: &str, _moe: bool, _num_experts: usize, lr_cfg: LrScheduleConfig) {
    let batches = load_batches(4);
    let mut cnn = SimpleCNN::new(10);

    let base_lr = 0.01f32;
    let mut hrm = Hrm::new(base_lr, 0.0);
    let scheduler: Box<dyn LearningRateSchedule> = match lr_cfg {
        LrScheduleConfig::Step { step_size, gamma } => {
            Box::new(StepLr::new(base_lr, step_size, gamma))
        }
        LrScheduleConfig::Cosine { max_steps } => Box::new(CosineLr::new(base_lr, max_steps)),
        LrScheduleConfig::Constant => Box::new(ConstantLr::new(base_lr)),
    };
    let epochs = 5;
    let mut step = 0usize;

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
                let lr = scheduler.next_lr(step);
                if opt == "hrm" {
                    hrm.lr = lr;
                    hrm.update(fc, bias, &grad_logits, &feat);
                } else {
                    let rows = fc.rows;
                    let cols = fc.cols;

                    // Compute outer product of `feat` and `grad_logits`
                    let mut grad_matrix = vec![0.0f32; rows * cols];
                    for (c, &g) in grad_logits.iter().enumerate() {
                        for (r, &f) in feat.iter().enumerate() {
                            grad_matrix[r * cols + c] = f * g; // mul
                        }
                    }

                    // Update weights in a single pass
                    for (w, &g) in fc.data.iter_mut().zip(grad_matrix.iter()) {
                        *w -= lr * g; // mul + add
                    }

                    // Update bias using slice-based subtraction
                    for (b, &g) in bias.iter_mut().zip(grad_logits.iter()) {
                        *b -= lr * g; // mul + add
                    }

                    // bias update + outer product + weight update
                    let ops = rows * cols * 3 + cols * 2;
                    math::inc_ops_by(ops);
                }
                step += 1;

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
