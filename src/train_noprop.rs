use crate::data::{load_pairs, to_matrix, Vocab};
use crate::encoder_t::EncoderT;
use crate::autograd::Tensor;
use crate::weights::save_model;

pub fn run() {
    let pairs = load_pairs();
    let vocab = Vocab::build();
    let vocab_size = vocab.itos.len();

    let model_dim = 64;
    let mut encoder = EncoderT::new(6, vocab_size, model_dim, 1, 256);
    let lr = 0.001;

    for epoch in 0..5 {
        for (src, tgt) in &pairs {
            let x = to_matrix(src, vocab_size);
            let enc_out = encoder.forward(&x);

            // noisy target
            let mut true_target = to_matrix(tgt, vocab_size);
            for v in &mut true_target.data {
                *v += (rand::random::<f32>() - 0.5) * 0.1;
            }
            let noisy = Tensor::from_matrix(true_target.clone(), false);

            let mut loss = 0.0;
            for (l_i, layer) in encoder.layers.iter_mut().enumerate() {
                let out = layer.forward(&enc_out);
                for i in 0..out.data.data.len() {
                    let d = out.data.data[i] - noisy.data.data[i];
                    loss += d * d;

                    // dummy update
                    let len = layer.attn.wq.w.data.data.len();
                    let idx = i % len;
                    layer.attn.wq.w.data.data[idx] -= lr * d;
                }
                println!("epoch {epoch} layer {l_i} noprop loss {loss}");
            }
        }
    }

    // Save trained encoder weights
    save_model("model.json", &encoder, None);
}
