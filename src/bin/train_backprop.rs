use std::env;

use indicatif::ProgressBar;
use vanillanoprop::data::load_batches;
use vanillanoprop::math::{self, Matrix};
use vanillanoprop::metrics::f1_score;
use vanillanoprop::models::{DecoderT, EncoderT};
use vanillanoprop::optim::{Adam, SGD};
use vanillanoprop::weights::save_model;

fn to_matrix(seq: &[u8], vocab_size: usize) -> Matrix {
    let mut m = Matrix::zeros(seq.len(), vocab_size);
    for (i, &tok) in seq.iter().enumerate() {
        m.set(i, tok as usize, 1.0);
    }
    m
}

fn main() {
    let opt = env::args().nth(1).unwrap_or_else(|| "sgd".to_string());
    run(&opt);
}

// Tensor Backprop Training (simplified Adam hook)
// now using Embedding => model_dim independent of vocab_size
fn run(opt: &str) {
    let batches = load_batches(4);
    let vocab_size = 256;

    let model_dim = 64;
    let mut encoder = EncoderT::new(6, vocab_size, model_dim, 256);
    let mut decoder = DecoderT::new(6, vocab_size, model_dim, 256);

    let lr = 0.001;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;
    let weight_decay = 0.0;
    let mut adam = Adam::new(lr, beta1, beta2, eps, weight_decay);
    let mut sgd = SGD::new(lr, weight_decay);

    math::reset_matrix_ops();
    let epochs = 50;
    let pb = ProgressBar::new(epochs as u64);
    let mut best_f1 = f32::NEG_INFINITY;
    for epoch in 0..epochs {
        let mut last_loss = 0.0;
        let mut f1_sum = 0.0;
        let mut sample_cnt: f32 = 0.0;
        for batch in &batches {
            encoder.zero_grad();
            decoder.zero_grad();
            let mut batch_loss = 0.0f32;
            let mut batch_f1 = 0.0f32;
            for (src, tgt) in batch {
                let tgt = *tgt;
                let enc_x = to_matrix(src, vocab_size);
                let enc_out = encoder.forward_train(&enc_x);

                let dec_in = vec![tgt as u8];
                let dec_x = to_matrix(&dec_in, vocab_size);
                let logits = decoder.forward_train(&dec_x, &enc_out);

                let (loss, grad, preds) =
                    math::softmax_cross_entropy(&logits, &[tgt], 0);
                batch_loss += loss;

                let grad_enc = decoder.backward(&grad);
                encoder.backward(&grad_enc);
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
            {
                let dec_params = decoder.parameters();
                params.extend(dec_params);
            }
            if opt == "sgd" {
                sgd.step(&mut params);
            } else {
                adam.step(&mut params);
            }
            println!("loss {batch_loss:.4} f1 {batch_f1_avg:.4}");
        }
        let avg_f1 = f1_sum / if sample_cnt > 0.0 { sample_cnt } else { 1.0 };
        pb.set_message(format!("epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}"));
        pb.inc(1);

        if avg_f1 > best_f1 {
            println!("Checkpoint saved at epoch {epoch}: avg F1 improved to {avg_f1:.4}");
            best_f1 = avg_f1;
            save_model("checkpoint.json", &mut encoder, Some(&mut decoder));
        }
    }
    pb.finish_with_message("training done");

    println!("Total matrix ops: {}", math::matrix_ops_count());

    // Save trained weights
    save_model("model.json", &mut encoder, Some(&mut decoder));
}
