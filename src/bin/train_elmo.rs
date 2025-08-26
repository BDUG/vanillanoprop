use std::env;

use indicatif::ProgressBar;
use vanillanoprop::data::load_batches;
use vanillanoprop::layers::Activation;
use vanillanoprop::math::{self, Matrix};
use vanillanoprop::memory;
use vanillanoprop::metrics::f1_score;
use vanillanoprop::models::EncoderT;
use vanillanoprop::optim::Adam;
use vanillanoprop::train_cnn;
use vanillanoprop::weights::save_model;

fn main() {
    let model = env::args()
        .nth(1)
        .unwrap_or_else(|| "transformer".to_string());
    if model == "cnn" {
        train_cnn::run("sgd");
    } else {
        run();
    }
}

fn run() {
    let batches = load_batches(4);
    let vocab_size = 256;

    // With embedding → model_dim separate
    let model_dim = 64;
    let mut encoder = EncoderT::new(6, vocab_size, model_dim, 128, Activation::ReLU);
    let lr = 0.001;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;
    let weight_decay = 0.0;
    let mut optim = Adam::new(lr, beta1, beta2, eps, weight_decay);

    math::reset_matrix_ops();
    let epochs = 5;
    let pb = ProgressBar::new(epochs as u64);
    let mut best_f1 = f32::NEG_INFINITY;
    for epoch in 0..epochs {
        let mut last_loss = 0.0;
        let mut f1_sum = 0.0;
        let mut sample_cnt: f32 = 0.0;
        for batch in &batches {
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
                let f1 = f1_score(&preds, &[tgt]);
                batch_f1 += f1;
            }
            let bsz = batch.len() as f32;
            batch_loss /= bsz;
            let batch_f1_avg = batch_f1 / bsz;
            last_loss = batch_loss;
            f1_sum += batch_f1;
            sample_cnt += bsz;
            let mut params = encoder.parameters();
            optim.step(&mut params);
            println!("loss {batch_loss:.4} f1 {batch_f1_avg:.4}");
        }
        let avg_f1 = f1_sum / if sample_cnt > 0.0 { sample_cnt } else { 1.0 };
        pb.set_message(format!("epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}"));
        pb.inc(1);

        if avg_f1 > best_f1 {
            println!("Checkpoint saved at epoch {epoch}: avg F1 improved to {avg_f1:.4}");
            best_f1 = avg_f1;
            save_model("checkpoint.json", &mut encoder, None);
        }
    }
    pb.finish_with_message("training done");

    println!("Total matrix ops: {}", math::matrix_ops_count());
    let peak = memory::peak_memory_bytes();
    println!(
        "Max memory usage: {:.2} MB",
        peak as f64 / (1024.0 * 1024.0)
    );

    save_model("model.json", &mut encoder, None);
}
