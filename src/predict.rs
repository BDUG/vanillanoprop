use crate::autograd::Tensor;
use crate::data::load_pairs;
use crate::math::{self, Matrix};
use crate::models::{DecoderT, EncoderT};
use crate::weights::load_model;
use rand::Rng;

fn to_matrix(seq: &[u8], vocab_size: usize) -> Matrix {
    let mut m = Matrix::zeros(seq.len(), vocab_size);
    for (i, &tok) in seq.iter().enumerate() {
        m.set(i, tok as usize, 1.0);
    }
    m
}

pub fn run() {
    // pick a random image from the MNIST training pairs
    let pairs = load_pairs();
    let mut rng = rand::thread_rng();
    let idx = rng.gen_range(0..pairs.len());
    let (src, tgt) = &pairs[idx];

    let vocab_size = 256;

    let model_dim = 64;
    let mut encoder = EncoderT::new(6, vocab_size, model_dim, 256);
    let mut decoder = DecoderT::new(6, vocab_size, model_dim, 256);

    load_model("model.json", &mut encoder, &mut decoder);

    math::reset_matrix_ops();
    let enc_x = to_matrix(src, vocab_size);
    let enc_out = encoder.forward(&enc_x);

    let dec_in = vec![0u8];
    let dec_x = to_matrix(&dec_in, vocab_size);
    let logits = decoder.forward(&Tensor::from_matrix(dec_x), &enc_out);
    let probs = Tensor::softmax(&logits);

    let mut best_tok = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for t in 0..vocab_size {
        let p = probs.data.get(0, t);
        if p > best_val {
            best_val = p;
            best_tok = t;
        }
    }

    println!(
        "{{\"actual\":{}, \"prediction\":{}}}",
        tgt, best_tok
    );
    println!("Total matrix ops: {}", math::matrix_ops_count());
}
