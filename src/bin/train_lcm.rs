use indicatif::ProgressBar;
use vanillanoprop::config::Config;
use vanillanoprop::data::{DataLoader, Mnist};
use vanillanoprop::model::Model;
use vanillanoprop::models::LargeConceptModel;
use vanillanoprop::weights::save_lcm;

mod common;

fn main() {
    let args = common::init_logging();
    let (
        _model,
        _opt,
        _moe,
        _num_experts,
        _lr_cfg,
        _resume,
        _save_every,
        _checkpoint_dir,
        _log_dir,
        _experiment,
        _export_onnx,
        fine_tune,
        freeze_layers,
        _auto_ml,
        config,
        _,
    ) = common::parse_cli(args.into_iter().skip(1));
    let _ft = fine_tune.map(|model_id| {
        vanillanoprop::fine_tune::run(
            &model_id,
            config.hf_token.as_deref(),
            freeze_layers,
            |_, _| Ok(()),
        )
        .expect("fine-tune load failed")
    });
    run(&config);
}

fn run(config: &Config) {
    let mut model = LargeConceptModel::new(28 * 28, 128, 64, 10);
    let lr = 0.01f32;
    let l2 = 1e-4f32;
    let evaluator = Model::new();

    let pb = ProgressBar::new(config.epochs as u64);
    let mut loader = DataLoader::<Mnist>::new(config.batch_size, false, None);
    for epoch in 0..config.epochs {
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
                targets.push(*tgt);
            }
            let (batch_loss, preds) = model.train_batch(&images, &targets, lr, l2);
            let batch_f1 = evaluator.evaluate(&preds, &targets);
            last_loss = batch_loss;
            f1_sum += batch_f1;
            sample_cnt += 1.0;
        }
        let avg_f1 = f1_sum / if sample_cnt > 0.0 { sample_cnt } else { 1.0 };
        pb.set_message(format!("epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}"));
        log::info!("epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}");
        pb.inc(1);
    }
    pb.finish_with_message("training done");

    if let Err(e) = save_lcm("lcm.json", &model) {
        log::error!("Failed to save model: {e}");
    }
}
