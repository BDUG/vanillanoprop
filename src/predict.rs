use crate::data::load_pairs;
use crate::layers::Activation;
use crate::math::{self, Matrix};
use crate::models::{DecoderT, EncoderT};
use crate::tensor::Tensor;
use crate::weights::{load_cnn, load_model};
use rand::Rng;

fn to_matrix(seq: &[u8], vocab_size: usize) -> Matrix {
    let mut m = Matrix::zeros(seq.len(), vocab_size);
    for (i, &tok) in seq.iter().enumerate() {
        m.set(i, tok as usize, 1.0);
    }
    m
}

pub fn run(model: Option<&str>) {
    // pick a random image from the MNIST training pairs
    let pairs = load_pairs();
    let mut rng = rand::thread_rng();
    let idx = rng.gen_range(0..pairs.len());
    let (src, tgt) = &pairs[idx];

    match model.unwrap_or("cnn") {
        "transformer" => {
            let vocab_size = 256;
            let model_dim = 64;
            let mut encoder = EncoderT::new(6, vocab_size, model_dim, 256, Activation::ReLU, false);
            let mut decoder = DecoderT::new(6, vocab_size, model_dim, 256, Activation::ReLU, false);

            load_model("model.json", &mut encoder, &mut decoder);

            math::reset_matrix_ops();
            let enc_x = to_matrix(src, vocab_size);
            let enc_out = encoder.forward(&enc_x);

            // Average encoder activations across the sequence
            let mut avg = Matrix::zeros(1, enc_out.data.cols);
            for c in 0..enc_out.data.cols {
                let mut sum = 0f32;
                for r in 0..enc_out.data.rows {
                    sum += enc_out.data.get(r, c);
                }
                avg.set(0, c, sum / enc_out.data.rows as f32);
            }

            let probs = Tensor::softmax(&Tensor::from_matrix(avg));

            let mut best_tok = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for t in 0..probs.data.cols {
                let p = probs.data.get(0, t);
                if p > best_val {
                    best_val = p;
                    best_tok = t;
                }
            }

            println!("{{\"actual\":{}, \"prediction\":{}}}", tgt, best_tok);
            println!("Total matrix ops: {}", math::matrix_ops_count());
        }
        _ => {
            // default CNN
            let cnn = load_cnn("cnn.json", 10);
            let pred = cnn.predict(src);
            println!("{{\"actual\":{}, \"prediction\":{}}}", tgt, pred);
        }
    }
}
