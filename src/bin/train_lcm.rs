use std::env;

use indicatif::ProgressBar;
use vanillanoprop::config::Config;
use vanillanoprop::data::load_batches;
use vanillanoprop::model::Model;
use vanillanoprop::models::LargeConceptModel;
use vanillanoprop::weights::save_lcm;

mod common;

fn main() {
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
        config,
        _,
    ) = common::parse_cli(env::args().skip(1));
    run(&config);
}

fn run(config: &Config) {
    let batches = load_batches(config.batch_size);
    let mut model = LargeConceptModel::new(28 * 28, 128, 10);
    let lr = 0.01f32;
    let evaluator = Model::new();

    let pb = ProgressBar::new(config.epochs as u64);
    for epoch in 0..config.epochs {
        let mut last_loss = 0.0f32;
        let mut f1_sum = 0.0f32;
        let mut sample_cnt = 0.0f32;
        for batch in &batches {
            let mut batch_loss = 0.0f32;
            let mut batch_f1 = 0.0f32;
            for (img, tgt) in batch {
                let (loss, pred) = model.train_step(img, *tgt, lr);
                batch_loss += loss;
                batch_f1 += evaluator.evaluate(&[pred], &[*tgt]);
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
    }
    pb.finish_with_message("training done");

    if let Err(e) = save_lcm("lcm.json", &model) {
        eprintln!("Failed to save model: {e}");
    }
}
