use indicatif::ProgressBar;
use vanillanoprop::config::Config;
use vanillanoprop::data::{Cifar10, DataLoader, Dataset, DatasetKind, Mnist};
use vanillanoprop::fine_tune::LayerKind;
use vanillanoprop::layers::{Activation, LinearT};
use vanillanoprop::logging::{Callback, Logger, MetricRecord};
use vanillanoprop::math::{self, Matrix};
use vanillanoprop::memory;
use vanillanoprop::model::Model;
use vanillanoprop::models::{DecoderT, EncoderT};
use vanillanoprop::optim::{Adam, MseLoss, SGD};
use vanillanoprop::train_cnn;
use vanillanoprop::util::logging::{log_checkpoint_saved, log_total_ops};

#[path = "common.rs"]
mod common;

fn main() {
    let args = common::init_logging();
    let (
        model,
        opt,
        moe,
        num_experts,
        lr_cfg,
        resume,
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
    ) = common::parse_cli(args.into_iter().skip(1));
    let ft = fine_tune.map(|model_id| {
        vanillanoprop::fine_tune::run(&model_id, freeze_layers, |_, _| Ok(()))
            .expect("fine-tune load failed")
    });
    if model == "cnn" || model == "mobilenet" {
        train_cnn::run(
            &model,
            &opt,
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
        run(
            DatasetKind::Mnist,
            &opt,
            moe,
            num_experts,
            log_dir,
            experiment,
            &config,
            resume,
            ft,
        );
    }
}

// Tensor Backprop Training (simplified Adam hook)
// now using Embedding => model_dim independent of vocab_size
fn run_impl<D: Dataset<Item = (Vec<u8>, usize)>>( 
    opt: &str,
    moe: bool,
    num_experts: usize,
    log_dir: Option<String>,
    experiment: Option<String>,
    config: &Config,
    resume: Option<String>,
    fine_tune: Option<vanillanoprop::fine_tune::FineTune>,
) {
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
    let mut decoder = DecoderT::new(
        6,
        vocab_size,
        model_dim,
        256,
        Activation::ReLU,
        moe,
        num_experts,
    );

    let lr = 0.001;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;
    let weight_decay = 0.0;
    let mut trainer = if let Some(path) = resume.as_deref() {
        let mut params = encoder.parameters();
        {
            let dec_params = decoder.parameters();
            params.extend(dec_params);
        }
        match Model::load(path, &mut params) {
            Ok(m) => {
                log::info!("Resumed model from {path}");
                m
            }
            Err(e) => {
                log::error!("Failed to load model {path}: {e}");
                Model::new()
            }
        }
    } else {
        Model::new()
    };
    if opt == "sgd" {
        trainer.compile(SGD::new(lr, weight_decay), MseLoss::new());
    } else {
        trainer.compile(
            Adam::new(lr, beta1, beta2, eps, weight_decay),
            MseLoss::new(),
        );
    }

    let mut logger = Logger::new(log_dir, experiment).ok();
    math::reset_matrix_ops();
    let epochs = config.epochs;
    let pb = ProgressBar::new(epochs as u64);
    let mut best_f1 = f32::NEG_INFINITY;
    let mut step = 0usize;
    let mut loader = DataLoader::<D>::new(config.batch_size, false, None);
    for epoch in 0..epochs {
        loader.reset(true);
        let mut last_loss = 0.0;
        let mut f1_sum = 0.0;
        let mut sample_cnt: f32 = 0.0;
        for batch in loader.by_ref() {
            encoder.zero_grad();
            decoder.zero_grad();
            let mut batch_loss = 0.0f32;
            let mut batch_f1 = 0.0f32;
            for (src, tgt) in batch {
                let tgt = *tgt;
                // one-hot encode source sequence for embedding layer
                let mut enc_x = Matrix::zeros(src.len(), vocab_size);
                for (i, &tok) in src.iter().enumerate() {
                    enc_x.set(i, tok as usize, 1.0);
                }
                let enc_out = encoder.forward_train(&enc_x);

                // one-hot encode target token for decoder input
                let mut dec_x = Matrix::zeros(1, vocab_size);
                dec_x.set(0, tgt as usize, 1.0);
                let logits = decoder.forward_train(&dec_x, &enc_out);

                let (loss, grad, preds) = math::softmax_cross_entropy(&logits, &[tgt], 0);
                batch_loss += loss;

                let grad_enc = decoder.backward(&grad);
                encoder.backward(&grad_enc);
                let f1 = trainer.evaluate(&preds, &[tgt]);
                batch_f1 += f1;
            }
            let bsz = batch.len() as f32;
            batch_loss /= bsz;
            let batch_f1_avg = batch_f1 / bsz;
            last_loss = batch_loss;
            f1_sum += batch_f1;
            sample_cnt += bsz;
            let mut params: Vec<(LayerKind, &mut LinearT)> = encoder
                .parameters()
                .into_iter()
                .map(|p| (LayerKind::Linear, p))
                .collect();
            {
                let dec_params = decoder.parameters();
                params.extend(dec_params.into_iter().map(|p| (LayerKind::Linear, p)));
            }
            if let Some(ft) = &fine_tune {
                let mut filtered = ft.filter(params);
                trainer.fit(&mut filtered);
            } else {
                let mut raw: Vec<&mut LinearT> = params.into_iter().map(|(_, p)| p).collect();
                trainer.fit(&mut raw);
            }
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
        log::info!("epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}");
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
            log_checkpoint_saved(epoch, avg_f1);
            best_f1 = avg_f1;
            let mut params = encoder.parameters();
            {
                let dec_params = decoder.parameters();
                params.extend(dec_params);
            }
            let param_refs: Vec<&LinearT> = params.iter().map(|p| &**p).collect();
            if let Err(e) = trainer.save("checkpoint.bin", &param_refs) {
                log::error!("Failed to save checkpoint: {e}");
            }
        }
    }
    pb.finish_with_message("training done");

    log_total_ops(math::matrix_ops_count());
    let peak = memory::peak_memory_bytes();
    log::info!(
        "Max memory usage: {:.2} MB",
        peak as f64 / (1024.0 * 1024.0)
    );

    // Save trained weights
    let mut params = encoder.parameters();
    {
        let dec_params = decoder.parameters();
        params.extend(dec_params);
    }
    let param_refs: Vec<&LinearT> = params.iter().map(|p| &**p).collect();
    if let Err(e) = trainer.save("model.bin", &param_refs) {
        log::error!("Failed to save model: {e}");
    }
}

pub fn run(
    dataset: DatasetKind,
    opt: &str,
    moe: bool,
    num_experts: usize,
    log_dir: Option<String>,
    experiment: Option<String>,
    config: &Config,
    resume: Option<String>,
    fine_tune: Option<vanillanoprop::fine_tune::FineTune>,
) {
    match dataset {
        DatasetKind::Mnist => run_impl::<Mnist>(
            opt,
            moe,
            num_experts,
            log_dir,
            experiment,
            config,
            resume,
            fine_tune,
        ),
        DatasetKind::Cifar10 => run_impl::<Cifar10>(
            opt,
            moe,
            num_experts,
            log_dir,
            experiment,
            config,
            resume,
            fine_tune,
        ),
    }
}
