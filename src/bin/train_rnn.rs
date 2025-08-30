use std::env;

use indicatif::ProgressBar;
use vanillanoprop::config::Config;
use vanillanoprop::data::{DataLoader, Mnist};
use vanillanoprop::fine_tune::LayerKind;
use vanillanoprop::layers::LinearT;
use vanillanoprop::logging::{Logger, MetricRecord};
use vanillanoprop::math::{self, Matrix};
use vanillanoprop::model::Model;
use vanillanoprop::models::RNN;
use vanillanoprop::optim::{Adam, MseLoss};
use vanillanoprop::weights::save_rnn;

mod common;

fn main() {
    env_logger::init();
    let (
        _model,
        _opt,
        _moe,
        _num_experts,
        _lr_cfg,
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

    let ft = fine_tune.map(|model_id| {
        vanillanoprop::fine_tune::run(&model_id, freeze_layers, |_, _| Ok(()))
            .expect("fine-tune load failed")
    });

    run(log_dir, experiment, &config, ft);
}

fn run(
    log_dir: Option<String>,
    experiment: Option<String>,
    config: &Config,
    fine_tune: Option<vanillanoprop::fine_tune::FineTune>,
) {
    let vocab_size = 256; // pixel values 0-255
    let hidden_dim = 64;
    let num_classes = 10;
    let mut rnn = RNN::new_gru(vocab_size, hidden_dim, num_classes);

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
    let epochs = config.epochs;
    let pb = ProgressBar::new(epochs as u64);
    let mut step = 0usize;

    for epoch in 0..epochs {
        let mut last_loss = 0.0;
        for batch in DataLoader::<Mnist>::new(config.batch_size, true, None) {
            rnn.zero_grad();
            let mut batch_loss = 0.0f32;
            let mut batch_f1 = 0.0f32;
            for (src, tgt) in batch {
                // one-hot encode image sequence
                let mut x = Matrix::zeros(src.len(), vocab_size);
                for (i, &tok) in src.iter().enumerate() {
                    x.set(i, tok as usize, 1.0);
                }
                let logits = rnn.forward_train(&x);
                let (loss, grad, preds) = math::softmax_cross_entropy(&logits, &[*tgt], 0);
                batch_loss += loss;
                rnn.backward(&grad);
                let f1 = trainer.evaluate(&preds, &[*tgt]);
                batch_f1 += f1;
            }
            let bsz = batch.len() as f32;
            batch_loss /= bsz;
            batch_f1 /= bsz;
            last_loss = batch_loss;
            let params: Vec<(LayerKind, &mut LinearT)> = rnn
                .parameters()
                .into_iter()
                .map(|p| (LayerKind::Linear, p))
                .collect();
            if let Some(ft) = &fine_tune {
                let mut filtered = ft.filter(params);
                trainer.fit(&mut filtered);
            } else {
                let mut raw: Vec<&mut LinearT> = params.into_iter().map(|(_, p)| p).collect();
                trainer.fit(&mut raw);
            }
            println!("loss {batch_loss:.4} f1 {batch_f1:.4}");
            if let Some(l) = &mut logger {
                l.log(&MetricRecord {
                    epoch,
                    step,
                    loss: batch_loss,
                    f1: batch_f1,
                    lr,
                    kind: "batch",
                });
            }
            step += 1;
        }
        pb.set_message(format!("epoch {epoch} loss {last_loss:.4}"));
        if let Some(l) = &mut logger {
            l.log(&MetricRecord {
                epoch,
                step,
                loss: last_loss,
                f1: 0.0,
                lr,
                kind: "epoch",
            });
        }
        pb.inc(1);
    }
    pb.finish_with_message("training done");

    if let Err(e) = save_rnn("rnn.json", &mut rnn) {
        eprintln!("Failed to save model: {e}");
    }
}
