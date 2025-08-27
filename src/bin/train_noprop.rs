use std::env;

use indicatif::ProgressBar;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use vanillanoprop::config::Config;
use vanillanoprop::data::load_batches;
use vanillanoprop::layers::Activation;
use vanillanoprop::logging::{Logger, MetricRecord};
use vanillanoprop::math::{self, Matrix};
use vanillanoprop::memory;
use vanillanoprop::metrics::f1_score;
use vanillanoprop::models::EncoderT;
use vanillanoprop::optim::lr_scheduler::{
    ConstantLr, CosineLr, LearningRateSchedule, LrScheduleConfig, StepLr,
};
use vanillanoprop::rng::rng_from_env;
use vanillanoprop::tensor::Tensor;
use vanillanoprop::train_cnn;
use vanillanoprop::weights::{
    load_checkpoint, save_checkpoint, save_model, tensor_to_vec2, vec2_to_matrix, ModelJson,
};

mod common;

fn main() {
    let (
        model,
        opt,
        moe,
        num_experts,
        lr_cfg,
        resume,
        save_every,
        checkpoint_dir,
        log_dir,
        experiment,
        config,
        _,
    ) = common::parse_cli(env::args().skip(1));
    if model == "cnn" {
        train_cnn::run(
            &opt,
            moe,
            num_experts,
            lr_cfg,
            resume,
            save_every,
            checkpoint_dir,
            log_dir,
            experiment,
            &config,
        );
    } else {
        run(
            moe,
            num_experts,
            lr_cfg,
            resume,
            save_every,
            checkpoint_dir,
            log_dir,
            experiment,
            &config,
        );
    }
}

#[derive(Serialize, Deserialize)]
struct NopropCheckpoint {
    epoch: usize,
    step: usize,
    best_f1: f32,
    model: ModelJson,
}

fn run(
    moe: bool,
    num_experts: usize,
    lr_cfg: LrScheduleConfig,
    resume: Option<String>,
    save_every: Option<usize>,
    checkpoint_dir: Option<String>,
    log_dir: Option<String>,
    experiment_name: Option<String>,
    config: &Config,
) {
    let batches = load_batches(config.batch_size);
    let mut rng = rng_from_env();
    let vocab_size = 256;

    let model_dim = 64;
    let mut encoder = EncoderT::new(
        6,
        vocab_size,
        model_dim,
        256,
        Activation::ReLU,
        moe,
        num_experts,
    );
    let base_lr = 0.001;
    let scheduler: Box<dyn LearningRateSchedule> = match lr_cfg {
        LrScheduleConfig::Step { step_size, gamma } => {
            Box::new(StepLr::new(base_lr, step_size, gamma))
        }
        LrScheduleConfig::Cosine { max_steps } => Box::new(CosineLr::new(base_lr, max_steps)),
        LrScheduleConfig::Constant => Box::new(ConstantLr::new(base_lr)),
    };
    let mut step = 0usize;
    let mut start_epoch = 0usize;
    let mut best_f1 = f32::NEG_INFINITY;

    if let Some(path) = &resume {
        if let Ok(cp) = load_checkpoint::<NopropCheckpoint>(path) {
            if !cp.model.encoder_embedding.is_empty() {
                let mut params = encoder.embedding.parameters();
                let exp_rows = params[0].w.data.rows;
                let exp_cols = params[0].w.data.cols;
                let loaded = vec2_to_matrix(&cp.model.encoder_embedding);
                let mut mat = Matrix::zeros(exp_rows, exp_cols);
                for r in 0..loaded.rows.min(exp_rows) {
                    for c in 0..loaded.cols.min(exp_cols) {
                        mat.set(r, c, loaded.get(r, c));
                    }
                }
                params[0].w = Tensor::from_matrix(mat);
            }
            step = cp.step;
            start_epoch = cp.epoch + 1;
            best_f1 = cp.best_f1;
        }
    }

    let ckpt_dir = checkpoint_dir.unwrap_or_else(|| {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        format!("runs/{ts}")
    });

    let mut logger = Logger::new(log_dir, experiment_name).ok();
    let mut last_lr = base_lr;

    math::reset_matrix_ops();
    let pb = ProgressBar::new(config.epochs as u64);
    pb.set_position(start_epoch as u64);
    for epoch in start_epoch..config.epochs {
        let mut last_loss = 0.0;
        let mut f1_sum = 0.0;
        let mut sample_cnt: f32 = 0.0;
        for batch in &batches {
            let mut batch_loss = 0.0f32;
            let mut batch_f1 = 0.0f32;
            for (src, tgt) in batch {
                let tgt = *tgt;
                let len = 1usize;

                // one-hot encode the source token for the embedding layer
                let mut x = Matrix::zeros(len, vocab_size);
                for (i, &tok) in src[..len].iter().enumerate() {
                    x.set(i, tok as usize, 1.0);
                }
                let enc_out = encoder.forward_local(&x);

                // encode target without affecting gradients and add noise
                let mut tgt_mat = Matrix::zeros(1, vocab_size);
                tgt_mat.set(0, tgt as usize, 1.0);
                let mut noisy = encoder.forward(tgt_mat);
                for v in &mut noisy.data.data {
                    *v += (rng.gen::<f32>() - 0.5) * 0.1;
                }

                // Mean squared error and local feedback alignment update
                let mut delta = Matrix::zeros(enc_out.rows, enc_out.cols);
                let mut loss = 0.0f32;
                for i in 0..len * model_dim {
                    let d = enc_out.data[i] - noisy.data.data[i];
                    loss += d * d;
                    delta.data[i] = 2.0 * d;
                }
                let n = (len * model_dim) as f32;
                if n > 0.0 {
                    loss /= n;
                    for v in delta.data.iter_mut() {
                        *v /= n;
                    }
                }

                batch_loss += loss;
                let lr = scheduler.next_lr(step);
                last_lr = lr;
                encoder.fa_update(&delta, lr);
                step += 1;
                let src_slice: Vec<usize> = src[..len].iter().map(|&v| v as usize).collect();
                let f1 = f1_score(&src_slice, &[tgt]);
                batch_f1 += f1;
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
            let params = encoder.embedding.parameters();
            let enc_emb = tensor_to_vec2(&params[0].w);
            let model = ModelJson {
                encoder_embedding: enc_emb,
                decoder_embedding: Vec::new(),
            };
            let cp = NopropCheckpoint {
                epoch,
                step,
                best_f1,
                model,
            };
            let path = format!("{}/epoch_{}.json", ckpt_dir, epoch);
            if let Err(e) = save_checkpoint(&path, &cp) {
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

    if let Err(e) = save_model(&format!("{}/model.json", ckpt_dir), &mut encoder, None) {
        eprintln!("Failed to save model: {e}");
    }
}
