use crate::data::{load_pairs, to_matrix, Vocab};
use crate::math;
use crate::metrics::f1_score;
use crate::transformer_t::EncoderT;
use crate::weights::save_model;
use indicatif::ProgressBar;

pub fn run() {
    let pairs = load_pairs();
    let vocab = Vocab::build();
    let vocab_size = vocab.itos.len();

    // With embedding → model_dim separate
    let model_dim = 64;
    let mut encoder = EncoderT::new(6, vocab_size, model_dim, 128);
    let lr = 0.001;

    math::reset_matrix_ops();
    let epochs = 5;
    let pb = ProgressBar::new(epochs as u64);
    let mut best_f1 = f32::NEG_INFINITY;
    for epoch in 0..epochs {
        let mut last_loss = 0.0;
        let mut f1_sum = 0.0;
        let mut sample_cnt: f32 = 0.0;
        for (src, tgt) in &pairs {
            let x = to_matrix(src, vocab_size);
            let out = encoder.forward(&x);

            // Cross Entropy
            let mut ce = 0.0;
            let mut cnt: f32 = 0.0;
            let mut preds = Vec::new();
            for (i, &_tok) in tgt.iter().enumerate() {
                if i >= out.data.rows {
                    break;
                }
                let mut sum = 0.0;
                let mut best_tok = 0;
                let mut best_val = f32::NEG_INFINITY;
                for t in 0..vocab_size {
                    let val = out.data.get(i, t);
                    sum += val.exp();
                    if val > best_val {
                        best_val = val;
                        best_tok = t;
                    }
                }
                let p = best_val.exp() / sum;
                ce += -(p + 1e-9).ln();
                cnt += 1.0;
                preds.push(best_tok);
            }
            let loss = ce / if cnt > 0.0 { cnt } else { 1.0 };
            last_loss = loss;

            let f1 = f1_score(&preds, tgt);
            f1_sum += f1;
            sample_cnt += 1.0;
            println!("loss {loss:.4} f1 {f1:.4}");

            // dummy weight update
            for layer in &mut encoder.layers {
                for w in &mut layer.attn.wq.w.data.data {
                    *w -= lr * loss;
                }
            }
        }
        let avg_f1 = f1_sum / if sample_cnt > 0.0 { sample_cnt } else { 1.0 };
        pb.set_message(format!("epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}"));
        pb.inc(1);

        if avg_f1 > best_f1 {
            println!("Checkpoint saved at epoch {epoch}: avg F1 improved to {avg_f1:.4}");
            best_f1 = avg_f1;
            save_model("checkpoint.json", &encoder, None);
        }
    }
    pb.finish_with_message("training done");

    println!("Total matrix ops: {}", math::matrix_ops_count());

    save_model("model.json", &encoder, None);
}
