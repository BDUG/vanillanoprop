use crate::data::{load_pairs, to_matrix, Vocab, START};
use crate::math::{self, Matrix};
use crate::metrics::f1_score;
use crate::transformer_t::{DecoderT, EncoderT};
use crate::weights::save_model;
use indicatif::ProgressBar;

// Tensor Backprop Training (simplified Adam hook)
// now using Embedding => model_dim independent of vocab_size
pub fn run(_opt: &str) {
    let pairs = load_pairs();
    let vocab = Vocab::build();
    let vocab_size = vocab.itos.len();

    let model_dim = 64;
    let mut encoder = EncoderT::new(6, vocab_size, model_dim, 256);
    let mut decoder = DecoderT::new(6, vocab_size, model_dim, 256);

    let lr = 0.001;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;
    let start_id = *vocab.stoi.get(START).unwrap();

    math::reset_matrix_ops();
    let epochs = 50;
    let pb = ProgressBar::new(epochs as u64);
    let mut best_f1 = f32::NEG_INFINITY;
    for epoch in 0..epochs {
        let mut last_loss = 0.0;
        let mut f1_sum = 0.0;
        let mut sample_cnt: f32 = 0.0;
        for (src, tgt) in &pairs {
            encoder.zero_grad();
            decoder.zero_grad();

            // Encode source sentence
            let enc_x = to_matrix(src, vocab_size);
            let enc_out = encoder.forward_train(&enc_x);

            // Decoder input uses teacher forcing (START + target[:-1])
            let mut dec_in = vec![start_id];
            dec_in.extend_from_slice(tgt);
            let dec_x = to_matrix(&dec_in, vocab_size);
            let logits = decoder.forward_train(&dec_x, &enc_out);

            let probs = logits.softmax();
            let mut grad = Matrix::zeros(logits.rows, logits.cols);
            let mut loss = 0.0f32;
            let mut preds = Vec::new();
            for (i, &tok) in tgt.iter().enumerate() {
                let row = i + 1;
                if row >= logits.rows {
                    break;
                }
                let mut best_tok = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for t in 0..vocab_size {
                    let p = probs.get(row, t);
                    grad.set(row, t, p);
                    if p > best_val {
                        best_val = p;
                        best_tok = t;
                    }
                }
                let p = probs.get(row, tok);
                loss += -(p + 1e-9).ln();
                grad.set(row, tok, grad.get(row, tok) - 1.0);
                preds.push(best_tok);
            }
            let cnt = tgt.len() as f32;
            if cnt > 0.0 {
                loss /= cnt;
            }
            for v in grad.data.iter_mut() {
                *v /= cnt.max(1.0);
            }
            last_loss = loss;

            // Backward through decoder and encoder
            let grad_enc = decoder.backward(&grad);
            encoder.backward(&grad_enc);
            decoder.adam_step(lr, beta1, beta2, eps);
            encoder.adam_step(lr, beta1, beta2, eps);

            let f1 = f1_score(&preds, tgt);
            f1_sum += f1;
            sample_cnt += 1.0;
            println!("loss {loss:.4} f1 {f1:.4}");
        }
        let avg_f1 = f1_sum / if sample_cnt > 0.0 { sample_cnt } else { 1.0 };
        pb.set_message(format!("epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}"));
        pb.inc(1);

        if avg_f1 > best_f1 {
            println!("Checkpoint saved at epoch {epoch}: avg F1 improved to {avg_f1:.4}");
            best_f1 = avg_f1;
            save_model("checkpoint.json", &encoder, Some(&decoder));
        }
    }
    pb.finish_with_message("training done");

    println!("Total matrix ops: {}", math::matrix_ops_count());

    // Save trained weights
    save_model("model.json", &encoder, Some(&decoder));
}
