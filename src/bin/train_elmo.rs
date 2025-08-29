use std::env;

use indicatif::ProgressBar;
use vanillanoprop::config::Config;
use vanillanoprop::data::{DataLoader, Mnist};
use vanillanoprop::layers::Activation;
use vanillanoprop::logging::{Callback, Logger, MetricRecord};
use vanillanoprop::math::{self, Matrix};
use vanillanoprop::memory;
use vanillanoprop::model::Model;
use vanillanoprop::models::EncoderT;
use vanillanoprop::optim::{Adam, MseLoss};
use vanillanoprop::train_cnn;
use vanillanoprop::weights::save_model;

mod common;

fn main() {
    let (
        model,
        _opt,
        moe,
        num_experts,
        lr_cfg,
        _resume,
        _save_every,
        _ckpt_dir,
        log_dir,
        experiment,
        _export_onnx,
        fine_tune,
        freeze_layers,
        _auto_ml,
        config,
        _,
    ) = common::parse_cli(env::args().skip(1));
    let _ft = fine_tune.map(|model_id| {
        vanillanoprop::fine_tune::run(&model_id, freeze_layers, |_, _| Ok(()))
            .expect("fine-tune load failed")
    });
    if model == "cnn" {
        train_cnn::run(
            "sgd",
            moe,
            num_experts,
            lr_cfg,
            None,
            None,
            None,
            log_dir,
            experiment,
            &config,
            Vec::<Box<dyn Callback>>::new(),
        );
    } else {
        run(moe, num_experts, log_dir, experiment, &config);
    }
}

fn run(
    moe: bool,
    num_experts: usize,
    log_dir: Option<String>,
    experiment: Option<String>,
    config: &Config,
) {
    let vocab_size = 256;

    // With embedding → model_dim separate
    let model_dim = 64;
    let mut encoder = EncoderT::new(
        6,
        vocab_size,
        model_dim,
        128,
        Activation::ReLU,
        moe,
        num_experts,
    );
    let lr = 0.001;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;
    let weight_decay = 0.0;
    let mut trainer = Model::new();
    trainer.compile(
        Adam::new(lr, beta1, beta2, eps, weight_decay),
        MseLoss::new(),
    );

    let mut logger = Logger::new(log_dir, experiment).ok();
    math::reset_matrix_ops();
    let epochs = config.epochs;
    let pb = ProgressBar::new(epochs as u64);
    let mut best_f1 = f32::NEG_INFINITY;
    let mut step = 0usize;
    for epoch in 0..epochs {
        let mut last_loss = 0.0;
        let mut f1_sum = 0.0;
        let mut sample_cnt: f32 = 0.0;
        for batch in DataLoader::<Mnist>::new(config.batch_size, true, None) {
            encoder.zero_grad();
            let mut batch_loss = 0.0f32;
            let mut batch_f1 = 0.0f32;
            for (src, tgt) in batch {
                let tgt = *tgt;
                // one-hot encode the input sequence for the embedding layer
                let mut x = Matrix::zeros(src.len(), vocab_size);
                for (i, &tok) in src.iter().enumerate() {
                    x.set(i, tok as usize, 1.0);
                }
                let logits = encoder.forward_train(&x);
                let (loss, grad, preds) = math::softmax_cross_entropy(&logits, &[tgt], 0);
                batch_loss += loss;
                encoder.backward(&grad);
                let f1 = trainer.evaluate(&preds, &[tgt]);
                batch_f1 += f1;
            }
            let bsz = batch.len() as f32;
            batch_loss /= bsz;
            let batch_f1_avg = batch_f1 / bsz;
            last_loss = batch_loss;
            f1_sum += batch_f1;
            sample_cnt += bsz;
            let mut params = encoder.parameters();
            trainer.fit(&mut params);
            println!("loss {batch_loss:.4} f1 {batch_f1_avg:.4}");
            if let Some(l) = &mut logger {
                l.log(&MetricRecord {
                    epoch,
                    step,
                    loss: batch_loss,
                    f1: batch_f1_avg,
                    lr,
                    kind: "batch",
                });
            }
            step += 1;
        }
        let avg_f1 = f1_sum / if sample_cnt > 0.0 { sample_cnt } else { 1.0 };
        pb.set_message(format!("epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}"));
        if let Some(l) = &mut logger {
            l.log(&MetricRecord {
                epoch,
                step,
                loss: last_loss,
                f1: avg_f1,
                lr,
                kind: "epoch",
            });
        }
        pb.inc(1);

        if avg_f1 > best_f1 {
            println!("Checkpoint saved at epoch {epoch}: avg F1 improved to {avg_f1:.4}");
            best_f1 = avg_f1;
            if let Err(e) = save_model("checkpoint.json", &mut encoder, None) {
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

    if let Err(e) = save_model("model.json", &mut encoder, None) {
        eprintln!("Failed to save model: {e}");
    }
}
