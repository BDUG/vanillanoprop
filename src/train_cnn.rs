use indicatif::ProgressBar;

use crate::config::Config;
use crate::data::{DataLoader, Mnist};
use crate::flow_matching::FlowModel;
use crate::layers::{Layer, LinearT, MixtureOfExpertsT};
use crate::logging::{Callback, CallbackSignal, Logger, MetricRecord};
use crate::math::{self, Matrix};
use crate::memory;
use crate::metrics::f1_score;
use crate::models::{HybridRnnTransformer, SimpleCNN};
use crate::optim::lr_scheduler::{
    ConstantLr, CosineLr, LearningRateSchedule, LrScheduleConfig, StepLr,
};
use crate::optim::Hrm;
use crate::util::logging::{log_checkpoint_saved, log_total_ops};
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
    let mut cnn = SimpleCNN::new(10);

    // Hyperparameters for hybrid RNN + Transformer block
    let rnn_hidden_dim = 32usize;
    let num_heads = 2usize;
    let ff_hidden = 64usize;
    // Instantiate the hybrid block (unused in the CNN training loop but
    // demonstrates configuration of the sequence model components).
    let _hybrid = HybridRnnTransformer::new(28 * 28, rnn_hidden_dim, 32, num_heads, ff_hidden, 0.1);
    let mut moe_layer = if moe {
        let n = num_experts.max(1);
        let experts: Vec<Box<dyn Layer>> = (0..n)
            .map(|_| Box::new(LinearT::new(28 * 28, 10)) as Box<dyn Layer>)
            .collect();
        Some(MixtureOfExpertsT::new(28 * 28, experts, 1))
    } else {
        None
    };
    let flow_model = FlowModel::new(|_t, x: &[f32]| x.to_vec());

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

    let mut loader = DataLoader::<Mnist>::new(config.batch_size, false, None);

    for cb in callbacks.iter_mut() {
        cb.on_train_begin();
    }
    let mut stop_training = false;

    for epoch in start_epoch..epochs {
        for cb in callbacks.iter_mut() {
            cb.on_epoch_begin(epoch);
        }
        loader.reset(true);
        let mut last_loss = 0.0f32;
        let mut f1_sum = 0.0f32;
        let mut sample_cnt = 0.0f32;

        for batch in loader.by_ref() {
            let bsz = batch.len();
            let mut images = Vec::with_capacity(bsz);
            let mut targets = Vec::with_capacity(bsz);
            for (img, tgt) in batch.iter() {
                images.push(img.clone());
                targets.push(*tgt as usize);
            }

            let (feat_m, logits_fc) = cnn.forward_batch(&images);
            let flow_target = vec![0.0f32; logits_fc.cols];
            let mut flow_loss = 0.0f32;
            for r in 0..logits_fc.rows {
                let row = &logits_fc.data[r * logits_fc.cols..(r + 1) * logits_fc.cols];
                flow_loss += flow_model.time_loss(row, &flow_target, 0.0, 1.0, 10);
            }
            flow_loss /= bsz as f32;

            let logits = if let Some(moe) = &mut moe_layer {
                moe.forward_train(&feat_m)
            } else {
                logits_fc.clone()
            };
            let (ce_loss, grad_m, preds) = math::softmax_cross_entropy(&logits, &targets, 0);
            let loss = ce_loss + flow_loss;
            let mut batch_loss = loss;

            let lr = scheduler.next_lr(step);
            last_lr = lr;
            if let Some(moe) = &mut moe_layer {
                moe.zero_grad();
                moe.backward(&grad_m);
                for p in moe.parameters() {
                    p.sgd_step(lr, 0.0);
                }
            } else {
                let (fc, bias) = cnn.parameters_mut();
                if opt == "hrm" {
                    for i in 0..grad_m.rows {
                        let g_row = &grad_m.data[i * grad_m.cols..(i + 1) * grad_m.cols];
                        let f_row = &feat_m.data[i * feat_m.cols..(i + 1) * feat_m.cols];
                        hrm.lr = lr;
                        hrm.update(fc, bias, g_row, f_row);
                    }
                } else {
                    let grad_fc = Matrix::matmul(&feat_m.transpose(), &grad_m);
                    for (w, g) in fc.data.iter_mut().zip(grad_fc.data.iter()) {
                        *w -= lr * g;
                    }
                    let mut grad_bias = vec![0.0f32; grad_m.cols];
                    for r in 0..grad_m.rows {
                        for c in 0..grad_m.cols {
                            grad_bias[c] += grad_m.get(r, c);
                        }
                    }
                    for (b, g) in bias.iter_mut().zip(grad_bias.iter()) {
                        *b -= lr * g;
                    }
                }
            }
            step += 1;

            let batch_f1 = f1_score(&preds, &targets);
            last_loss = batch_loss;
            f1_sum += batch_f1;
            sample_cnt += 1.0;
            log::info!("loss {batch_loss:.4} f1 {batch_f1:.4}");
            let record = MetricRecord {
                epoch,
                step,
                loss: batch_loss,
                f1: batch_f1,
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
            log_checkpoint_saved(epoch, avg_f1);
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
                log::error!("Failed to save checkpoint: {e}");
            }
        }
    }

    for cb in callbacks.iter_mut() {
        cb.on_train_end();
    }

    pb.finish_with_message("training done");

    log_total_ops(math::matrix_ops_count());
    let peak = memory::peak_memory_bytes();
    log::info!(
        "Max memory usage: {:.2} MB",
        peak as f64 / (1024.0 * 1024.0)
    );
    if let Err(e) = save_cnn(&format!("{}/cnn.json", ckpt_dir), &cnn) {
        log::error!("Failed to save model: {e}");
    }
}
