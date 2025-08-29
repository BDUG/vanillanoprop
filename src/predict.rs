use crate::data::{DataLoader, Mnist};
use crate::layers::{Activation, Layer, LinearT, MixtureOfExpertsT};
use crate::math::{self, Matrix};
use crate::models::{DecoderT, EncoderT, LargeConceptModel, RnnCell, SimpleCNN, RNN};
use crate::tensor::Tensor;
use crate::weights::{load_cnn, load_lcm, load_model, load_moe, load_rnn};
use rand::{thread_rng, Rng};

fn to_matrix(seq: &[u8], vocab_size: usize) -> Matrix {
    let mut m = Matrix::zeros(seq.len(), vocab_size);
    for (i, &tok) in seq.iter().enumerate() {
        m.set(i, tok as usize, 1.0);
    }
    m
}

pub fn run(model: Option<&str>, moe: bool, num_experts: usize) {
    // pick a random image from the MNIST training pairs
    let pairs: Vec<(Vec<u8>, usize)> = DataLoader::<Mnist>::new(1, true, None)
        .flat_map(|b| b.iter().cloned())
        .collect();
    let mut rng = thread_rng();
    let idx = rng.gen_range(0..pairs.len());
    let (src, tgt) = &pairs[idx];

    match model.unwrap_or("cnn") {
        "transformer" => {
            let vocab_size = 256;
            let model_dim = 64;
            let mut encoder = EncoderT::new(
                6,
                vocab_size,
                model_dim,
                256,
                Activation::ReLU,
                moe,
                num_experts,
            );
            let mut decoder = DecoderT::new(
                6,
                vocab_size,
                model_dim,
                256,
                Activation::ReLU,
                moe,
                num_experts,
            );

            if let Err(e) = load_model("model.json", &mut encoder, &mut decoder) {
                eprintln!("Failed to load model weights: {e}");
            }

            math::reset_matrix_ops();
            let enc_x = to_matrix(src, vocab_size);
            let enc_out = encoder.forward(enc_x);

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
        "lcm" => {
            let model = match load_lcm("lcm.json", 28 * 28, 128, 64, 10) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Using random LCM weights; failed to load lcm.json: {e}");
                    LargeConceptModel::new(28 * 28, 128, 64, 10)
                }
            };
            if moe {
                let n = num_experts.max(1);
                let moe_layer = match load_moe("moe.json", 64, 10, n) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("Using random MoE weights; failed to load moe.json: {e}");
                        let experts: Vec<Box<dyn Layer>> = (0..n)
                            .map(|_| Box::new(LinearT::new(64, 10)) as Box<dyn Layer>)
                            .collect();
                        MixtureOfExpertsT::new(64, experts, 1)
                    }
                };
                let (feat, _logits) = model.forward(src);
                let feat_m = Matrix::from_vec(1, feat.len(), feat);
                let logits = moe_layer.forward(&Tensor::from_matrix(feat_m));
                let probs = Tensor::softmax(&logits);
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
            } else {
                let pred = model.predict(src);
                println!("{{\"actual\":{}, \"prediction\":{}}}", tgt, pred);
            }
        }
        "rnn" => {
            let vocab_size = 256;
            let hidden_dim = 64;
            let num_classes = 10;
            let model = match load_rnn("rnn.json", vocab_size, hidden_dim, num_classes) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Using random RNN weights; failed to load rnn.json: {e}");
                    RNN::new_gru(vocab_size, hidden_dim, num_classes)
                }
            };
            let enc_x = to_matrix(src, vocab_size);
            if moe {
                let n = num_experts.max(1);
                let moe_layer = match load_moe("moe.json", hidden_dim, num_classes, n) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("Using random MoE weights; failed to load moe.json: {e}");
                        let experts: Vec<Box<dyn Layer>> = (0..n)
                            .map(|_| {
                                Box::new(LinearT::new(hidden_dim, num_classes)) as Box<dyn Layer>
                            })
                            .collect();
                        MixtureOfExpertsT::new(hidden_dim, experts, 1)
                    }
                };
                let enc_x = Tensor::from_matrix(enc_x);
                let h = match &model.cell {
                    RnnCell::LSTM(l) => l.forward(&enc_x),
                    RnnCell::GRU(g) => g.forward(&enc_x),
                };
                let last_row = h.data.rows - 1;
                let mut last = Matrix::zeros(1, h.data.cols);
                for c in 0..h.data.cols {
                    last.set(0, c, h.data.get(last_row, c));
                }
                let logits = moe_layer.forward(&Tensor::from_matrix(last));
                let probs = Tensor::softmax(&logits);
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
            } else {
                let logits = model.forward(&Tensor::from_matrix(enc_x));
                let probs = Tensor::softmax(&logits);
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
            }
        }
        _ => {
            // default CNN
            let cnn = match load_cnn("cnn.json", 10) {
                Ok(cnn) => cnn,
                Err(e) => {
                    eprintln!("Using random CNN weights; failed to load cnn.json: {e}");
                    SimpleCNN::new(10)
                }
            };
            if moe {
                let n = num_experts.max(1);
                let moe_layer = match load_moe("moe.json", 28 * 28, 10, n) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("Using random MoE weights; failed to load moe.json: {e}");
                        let experts: Vec<Box<dyn Layer>> = (0..n)
                            .map(|_| Box::new(LinearT::new(28 * 28, 10)) as Box<dyn Layer>)
                            .collect();
                        MixtureOfExpertsT::new(28 * 28, experts, 1)
                    }
                };
                let (feat, _logits) = cnn.forward(src);
                let feat_m = Matrix::from_vec(1, feat.len(), feat);
                let logits = moe_layer.forward(&Tensor::from_matrix(feat_m));
                let probs = Tensor::softmax(&logits);
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
            } else {
                let pred = cnn.predict(src);
                println!("{{\"actual\":{}, \"prediction\":{}}}", tgt, pred);
            }
        }
    }
}
