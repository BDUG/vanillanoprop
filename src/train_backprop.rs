use crate::data::{load_pairs, to_matrix, Vocab, START, END};
use crate::transformer_t::{DecoderT, EncoderT};
use crate::autograd::Tensor;
use crate::weights::save_model;
use crate::metrics::f1_score;
use crate::math;
use indicatif::ProgressBar;

fn naive_decode(start_id: usize, end_id: usize) -> Vec<usize> {
    vec![start_id, end_id]
}

// Tensor Backprop Training (simplified Adam hook)
// now using Embedding => model_dim independent of vocab_size
pub fn run(_opt: &str) {
    let pairs = load_pairs();
    let vocab = Vocab::build();
    let vocab_size = vocab.itos.len();

    let model_dim = 64;
    let mut encoder = EncoderT::new(6, vocab_size, model_dim, 256);
    let decoder = DecoderT::new(6, vocab_size, model_dim, 256);

    let lr = 0.001;
    let start_id = *vocab.stoi.get(START).unwrap();
    let end_id = *vocab.stoi.get(END).unwrap();

    math::reset_matrix_ops();
    let epochs = 50;
    let pb = ProgressBar::new(epochs as u64);
    for epoch in 0..epochs {
        let mut last_loss = 0.0;
        let mut f1_sum = 0.0;
        let mut sample_cnt: f32 = 0.0;
        for (src, tgt) in &pairs {
            // Encode
            let enc_x = to_matrix(src, vocab_size);
            let enc_out = encoder.forward(&enc_x);

            // Naive decoding
            let generated = naive_decode(start_id, end_id);

            // CrossEntropy-Loss
            let loss = cross_entropy(&decoder, &enc_out, &generated, tgt, vocab_size);
            last_loss = loss;

            let f1 = f1_score(&generated, tgt);
            f1_sum += f1;
            sample_cnt += 1.0;
            println!("loss {loss:.4} f1 {f1:.4}");

            // Adam-Update Placeholder
            for layer in &mut encoder.layers {
                for w in &mut layer.attn.wq.w.data.data {
                    *w -= lr * loss;
                }
            }
        }
        let avg_f1 = f1_sum / if sample_cnt > 0.0 { sample_cnt } else { 1.0 };
        pb.set_message(format!("epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}"));
        pb.inc(1);
    }
    pb.finish_with_message("training done");

    println!("Total matrix ops: {}", math::matrix_ops_count());

    // Save trained weights
    save_model("model.json", &encoder, Some(&decoder));
}

fn cross_entropy(
    decoder: &DecoderT,
    enc_out: &Tensor,
    generated: &[usize],
    target: &[usize],
    vocab_size: usize,
) -> f32 {
    let mut ce = 0.0;
    let mut cnt = 0.0;
    for (i, &tok) in target.iter().enumerate() {
        if i >= generated.len() {
            break;
        }
        let t_in = to_matrix(&generated[..i + 1], vocab_size);
        let logits = decoder.forward(&Tensor::from_matrix(t_in), enc_out);
        let last = logits.data.rows - 1;
        let mut sum = 0.0;
        for t in 0..vocab_size {
            sum += logits.data.get(last, t).exp();
        }
        let p = logits.data.get(last, tok).exp() / sum;
        ce += -(p + 1e-9).ln();
        cnt += 1.0;
    }
    ce / cnt
}
