use crate::data::{load_mnist_pairs, to_matrix, Vocab};
use crate::transformer_t::EncoderT;
use crate::weights::save_model;
use indicatif::ProgressBar;

pub fn run() {
    let pairs = load_mnist_pairs();
    let vocab = Vocab::build_mnist();
    let vocab_size = vocab.itos.len();

    let model_dim = 64;
    let mut encoder = EncoderT::new(6, vocab_size, model_dim, 1, 256);
    let lr = 0.001;

    let epochs = 5;
    let pb = ProgressBar::new(epochs as u64);
    for epoch in 0..epochs {
        let mut last_loss = 0.0;
        for (src, tgt) in &pairs {
            let x = to_matrix(src, vocab_size);
            let enc_out = encoder.forward(&x);

            // encode target sentence with the same encoder to obtain a
            // comparable representation and add a bit of noise
            let mut noisy = encoder.forward(&to_matrix(tgt, vocab_size));
            for v in &mut noisy.data.data {
                *v += (rand::random::<f32>() - 0.5) * 0.1;
            }

            let mut loss = 0.0;
            for layer in encoder.layers.iter_mut() {
                let out = layer.forward(&enc_out);

                // ensure we only iterate over the common length to avoid
                // indexing past the end when dimensions mismatch
                let len = out.data.data.len().min(noisy.data.data.len());
                for i in 0..len {
                    let d = out.data.data[i] - noisy.data.data[i];
                    loss += d * d;

                    let w_len = layer.attn.wq.w.data.data.len();
                    let idx = i % w_len;
                    layer.attn.wq.w.data.data[idx] -= lr * d;
                }
            }
            last_loss = loss;
        }
        pb.set_message(format!("epoch {epoch} loss {last_loss:.4}"));
        pb.inc(1);
    }
    pb.finish_with_message("training done");

    save_model("model.json", &encoder, None);
}
