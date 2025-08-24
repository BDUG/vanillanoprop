use crate::data::{load_pairs, to_matrix, Vocab};
use crate::transformer_t::EncoderT;
use crate::weights::save_model;
use indicatif::ProgressBar;

pub fn run() {
    let pairs = load_pairs();
    let vocab = Vocab::build();
    let vocab_size = vocab.itos.len();

    // With embedding → model_dim separate
    let model_dim = 64;
    let mut encoder = EncoderT::new(6, vocab_size, model_dim, 1, 128);
    let lr = 0.001;

    let epochs = 5;
    let pb = ProgressBar::new(epochs as u64);
    for epoch in 0..epochs {
        let mut last_loss = 0.0;
        for (src, tgt) in &pairs {
            let x = to_matrix(src, vocab_size);
            let out = encoder.forward(&x);

            // Cross Entropy
            let mut ce = 0.0;
            let mut cnt = 0.0;
            for (i, &tok) in tgt.iter().enumerate() {
                if i >= out.data.rows {
                    break;
                }
                let mut sum = 0.0;
                for t in 0..vocab_size {
                    sum += out.data.get(i, t).exp();
                }
                let p = out.data.get(i, tok).exp() / sum;
                ce += -(p + 1e-9).ln();
                cnt += 1.0;
            }
            let loss = ce / cnt;
            last_loss = loss;

            // dummy weight update
            for layer in &mut encoder.layers {
                for w in &mut layer.attn.wq.w.data.data {
                    *w -= lr * loss;
                }
            }
        }
        pb.set_message(format!("epoch {epoch} loss {last_loss:.4}"));
        pb.inc(1);
    }
    pb.finish_with_message("training done");

    save_model("model.json", &encoder, None);
}
