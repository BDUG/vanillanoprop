use crate::data::{load_pairs, to_matrix, Vocab};
use crate::math::{self, Matrix};
use crate::metrics::f1_score;
use crate::transformer_t::EncoderT;
use crate::weights::save_model;
use indicatif::ProgressBar;

pub fn run() {
    let pairs = load_pairs();
    let vocab = Vocab::build();
    let vocab_size = vocab.itos.len();

    let model_dim = 64;
    let mut encoder = EncoderT::new(6, vocab_size, model_dim, 256);
    let lr = 0.001;
    let weight_decay = 0.0;

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
            let enc_out = encoder.forward_local(&x);

            // encode target without affecting gradients and add noise
            let mut noisy = encoder.forward(&to_matrix(tgt, vocab_size));
            for v in &mut noisy.data.data {
                *v += (rand::random::<f32>() - 0.5) * 0.1;
            }

            // Mean squared error and error signal
            let mut err = Matrix::zeros(enc_out.rows, enc_out.cols);
            let mut loss = 0.0f32;
            for i in 0..enc_out.data.len() {
                let d = enc_out.data[i] - noisy.data.data[i];
                loss += d * d;
                err.data[i] = d;
            }
            let n = enc_out.data.len() as f32;
            if n > 0.0 {
                loss /= n;
                for v in err.data.iter_mut() {
                    *v /= n;
                }
            }
            last_loss = loss;

            encoder.local_update(&err, lr, weight_decay);

            let f1 = f1_score(&src[..tgt.len().min(src.len())], tgt);
            f1_sum += f1;
            sample_cnt += 1.0;
            println!("loss {last_loss:.4} f1 {f1:.4}");
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
