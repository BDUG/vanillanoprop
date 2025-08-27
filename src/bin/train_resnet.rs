use std::env;

use indicatif::ProgressBar;
use vanillanoprop::config::Config;
use vanillanoprop::data::load_batches;
use vanillanoprop::logging::{Logger, MetricRecord};
use vanillanoprop::math::{self, Matrix};
use vanillanoprop::memory;
use vanillanoprop::metrics::f1_score;
use vanillanoprop::models::ResNet;
use vanillanoprop::optim::lr_scheduler::{
    ConstantLr, CosineLr, LearningRateSchedule, LrScheduleConfig, StepLr,
};
use vanillanoprop::optim::Hrm;

mod common;

fn main() {
    let (
        _model,
        opt,
        _moe,
        _num_experts,
        lr_cfg,
        _resume,
        _save_every,
        _ckpt_dir,
        log_dir,
        experiment,
        config,
        _,
    ) = common::parse_cli(env::args().skip(1));
    run(&opt, lr_cfg, log_dir, experiment, &config);
}

fn run(
    opt: &str,
    lr_cfg: LrScheduleConfig,
    log_dir: Option<String>,
    experiment_name: Option<String>,
    config: &Config,
) {
    let batches = load_batches(config.batch_size);
    // 64 hidden units and 2 residual blocks as a default configuration.
    let mut model = ResNet::new(10, 64, 2);

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
    let mut logger = Logger::new(log_dir, experiment_name).ok();
    let pb = ProgressBar::new(epochs as u64);
    let mut last_lr = base_lr;

    math::reset_matrix_ops();

    for epoch in 0..epochs {
        let mut last_loss = 0.0f32;
        let mut f1_sum = 0.0f32;
        let mut sample_cnt = 0.0f32;
        for batch in &batches {
            let mut batch_loss = 0.0f32;
            let mut batch_f1 = 0.0f32;
            for (img, tgt) in batch {
                let (feat, logits) = model.forward(img);
                let logits_m = Matrix::from_vec(1, logits.len(), logits);
                let (loss, grad_m, preds) =
                    math::softmax_cross_entropy(&logits_m, &[*tgt as usize], 0);
                batch_loss += loss;
                let grad_logits = grad_m.data;

                let (fc, bias) = model.parameters_mut();
                let lr = scheduler.next_lr(step);
                last_lr = lr;
                if opt == "hrm" {
                    hrm.lr = lr;
                    hrm.update(fc, bias, &grad_logits, &feat);
                } else {
                    let rows = fc.rows;
                    let cols = fc.cols;
                    let mut grad_matrix = vec![0.0f32; rows * cols];
                    for (c, &g) in grad_logits.iter().enumerate() {
                        for (r, &f) in feat.iter().enumerate() {
                            grad_matrix[r * cols + c] = f * g;
                        }
                    }
                    for (w, &g) in fc.data.iter_mut().zip(grad_matrix.iter()) {
                        *w -= lr * g;
                    }
                    for (b, &g) in bias.iter_mut().zip(grad_logits.iter()) {
                        *b -= lr * g;
                    }
                }

                let f1 = f1_score(&preds, &[*tgt as usize]);
                batch_f1 += f1;
                step += 1;
            }
            let bsz = batch.len() as f32;
            batch_loss /= bsz;
            let batch_f1_avg = batch_f1 / bsz;
            last_loss = batch_loss;
            f1_sum += batch_f1;
            sample_cnt += bsz;
            println!("loss {batch_loss:.4} f1 {batch_f1_avg:.4}");
            if let Some(l) = &mut logger {
                l.log(&MetricRecord {
                    epoch,
                    step,
                    loss: batch_loss,
                    f1: batch_f1_avg,
                    lr: last_lr,
                    kind: "batch",
                });
            }
        }
        let avg_f1 = f1_sum / if sample_cnt > 0.0 { sample_cnt } else { 1.0 };
        pb.set_message(format!("epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}"));
        if let Some(l) = &mut logger {
            l.log(&MetricRecord {
                epoch,
                step,
                loss: last_loss,
                f1: avg_f1,
                lr: last_lr,
                kind: "epoch",
            });
        }
        pb.inc(1);
    }

    pb.finish_with_message("training done");
    println!("Total matrix ops: {}", math::matrix_ops_count());
    let peak = memory::peak_memory_bytes();
    println!("Max memory usage: {:.2} MB", peak as f64 / (1024.0 * 1024.0));
}
