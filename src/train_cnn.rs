use indicatif::ProgressBar;

use crate::config::Config;
use crate::data::load_batches;
use crate::logging::{Callback, CallbackSignal, Logger, MetricRecord};
use crate::math::{self, Matrix};
use crate::memory;
use crate::metrics::f1_score;
use crate::models::SimpleCNN;
use crate::layers::{Layer, LinearT, MixtureOfExpertsT};
use crate::optim::lr_scheduler::{
    ConstantLr, CosineLr, LearningRateSchedule, LrScheduleConfig, StepLr,
};
use crate::optim::Hrm;
use crate::weights::{
    load_checkpoint, matrix_to_vec2, save_checkpoint, save_cnn, vec2_to_matrix, CnnJson,
};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Train a [`SimpleCNN`] on the MNIST data using a basic SGD loop.
///
/// `opt` selects the optimisation algorithm.  When `moe` is set a
/// [`MixtureOfExpertsT`] layer with `num_experts` experts replaces the
/// final fully connected layer.
#[derive(Serialize, Deserialize)]
struct CnnCheckpoint {
    epoch: usize,
    step: usize,
    best_f1: f32,
    model: CnnJson,
    hrm: Option<Hrm>,
}

pub fn run(
    opt: &str,
    moe: bool,
    num_experts: usize,
    lr_cfg: LrScheduleConfig,
    resume: Option<String>,
    save_every: Option<usize>,
    checkpoint_dir: Option<String>,
    log_dir: Option<String>,
    experiment_name: Option<String>,
    config: &Config,
    mut callbacks: Vec<Box<dyn Callback>>,
) {
    let batches = load_batches(config.batch_size);
    let mut cnn = SimpleCNN::new(10);
    let mut moe_layer = if moe {
        let n = num_experts.max(1);
        let experts: Vec<Box<dyn Layer>> = (0..n)
            .map(|_| Box::new(LinearT::new(28 * 28, 10)) as Box<dyn Layer>)
            .collect();
        Some(MixtureOfExpertsT::new(28 * 28, experts, 1))
    } else {
        None
    };

    let base_lr = 0.01f32;
    let mut hrm = Hrm::new(base_lr, 0.0);
    let scheduler: Box<dyn LearningRateSchedule> = match lr_cfg {
        LrScheduleConfig::Step { step_size, gamma } => {
            Box::new(StepLr::new(base_lr, step_size, gamma))
        }
        LrScheduleConfig::Cosine { max_steps } => Box::new(CosineLr::new(base_lr, max_steps)),
        LrScheduleConfig::Constant => Box::new(ConstantLr::new(base_lr)),
    };
    let epochs = config.epochs;
    let mut step = 0usize;
    let mut start_epoch = 0usize;
    let mut best_f1 = f32::NEG_INFINITY;

    if let Some(path) = &resume {
        if let Ok(cp) = load_checkpoint::<CnnCheckpoint>(path) {
            let (fc, bias) = cnn.parameters_mut();
            if !cp.model.fc.is_empty() {
                *fc = vec2_to_matrix(&cp.model.fc);
            }
            if !cp.model.bias.is_empty() {
                *bias = cp.model.bias.clone();
            }
            if let Some(h) = cp.hrm {
                hrm = h;
            }
            step = cp.step;
            start_epoch = cp.epoch + 1;
            best_f1 = cp.best_f1;
        }
    }

    let ckpt_dir = checkpoint_dir.unwrap_or_else(|| {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before UNIX_EPOCH")
            .as_secs();
        format!("runs/{ts}")
    });

    let mut logger = Logger::new(log_dir, experiment_name).ok();
    let mut last_lr = base_lr;

    math::reset_matrix_ops();
    let pb = ProgressBar::new(epochs as u64);
    pb.set_position(start_epoch as u64);

    for cb in callbacks.iter_mut() {
        cb.on_train_begin();
    }
    let mut stop_training = false;

    for epoch in start_epoch..epochs {
        for cb in callbacks.iter_mut() {
            cb.on_epoch_begin(epoch);
        }
        let mut last_loss = 0.0f32;
        let mut f1_sum = 0.0f32;
        let mut sample_cnt = 0.0f32;

        for batch in &batches {
            let mut batch_loss = 0.0f32;
            let mut batch_f1 = 0.0f32;

            for (img, tgt) in batch {
                let (feat, logits_fc) = cnn.forward(img);
                let (loss, grad_m, preds) = if let Some(moe) = &mut moe_layer {
                    let feat_m = Matrix::from_vec(1, feat.len(), feat.clone());
                    let logits_m = moe.forward_train(&feat_m);
                    math::softmax_cross_entropy(&logits_m, &[*tgt as usize], 0)
                } else {
                    let logits_m = Matrix::from_vec(1, logits_fc.len(), logits_fc);
                    math::softmax_cross_entropy(&logits_m, &[*tgt as usize], 0)
                };
                batch_loss += loss;

                let lr = scheduler.next_lr(step);
                last_lr = lr;
                if let Some(moe) = &mut moe_layer {
                    moe.zero_grad();
                    moe.backward(&grad_m);
                    for p in moe.parameters() {
                        if opt == "hrm" {
                            // HRM is not defined for MoE; fall back to SGD
                            p.sgd_step(lr, 0.0);
                        } else {
                            p.sgd_step(lr, 0.0);
                        }
                    }
                } else {
                    let grad_logits = grad_m.data;
                    // Update weights
                    let (fc, bias) = cnn.parameters_mut();
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
            let record = MetricRecord {
                epoch,
                step,
                loss: batch_loss,
                f1: batch_f1_avg,
                lr: last_lr,
                kind: "batch",
            };
            if let Some(l) = &mut logger {
                l.log(&record);
            }
            for cb in callbacks.iter_mut() {
                if let CallbackSignal::Stop = cb.on_batch_end(&record) {
                    stop_training = true;
                }
            }
            if stop_training {
                break;
            }
        }

        if stop_training {
            break;
        }

        let avg_f1 = f1_sum / if sample_cnt > 0.0 { sample_cnt } else { 1.0 };
        pb.set_message(format!("epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}"));
        let record = MetricRecord {
            epoch,
            step,
            loss: last_loss,
            f1: avg_f1,
            lr: last_lr,
            kind: "epoch",
        };
        if let Some(l) = &mut logger {
            l.log(&record);
        }
        for cb in callbacks.iter_mut() {
            if let CallbackSignal::Stop = cb.on_epoch_end(&record) {
                stop_training = true;
            }
        }
        pb.inc(1);
        if stop_training {
            break;
        }

        let mut should_save = false;
        if avg_f1 > best_f1 {
            println!("Checkpoint saved at epoch {epoch}: avg F1 improved to {avg_f1:.4}");
            best_f1 = avg_f1;
            should_save = true;
        }
        if let Some(n) = save_every {
            if (epoch + 1) % n == 0 {
                should_save = true;
            }
        }
        if should_save {
            let (fc, bias) = cnn.parameters();
            let model = CnnJson {
                fc: matrix_to_vec2(fc),
                bias: bias.clone(),
            };
            let cp = CnnCheckpoint {
                epoch,
                step,
                best_f1,
                model,
                hrm: if opt == "hrm" {
                    Some(hrm.clone())
                } else {
                    None
                },
            };
            let path = format!("{}/epoch_{}.json", ckpt_dir, epoch);
            if let Err(e) = save_checkpoint(&path, &cp) {
                eprintln!("Failed to save checkpoint: {e}");
            }
        }
    }

    for cb in callbacks.iter_mut() {
        cb.on_train_end();
    }

    pb.finish_with_message("training done");

    println!("Total matrix ops: {}", math::matrix_ops_count());
    let peak = memory::peak_memory_bytes();
    println!(
        "Max memory usage: {:.2} MB",
        peak as f64 / (1024.0 * 1024.0)
    );
    if let Err(e) = save_cnn(&format!("{}/cnn.json", ckpt_dir), &cnn) {
        eprintln!("Failed to save model: {e}");
    }
}
