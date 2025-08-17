use crate::data::{load_pairs, to_matrix, Vocab, START, END};
use crate::encoder_t::EncoderT;
use crate::decoder_t::DecoderT;
use crate::autograd::Tensor;
use serde_json::json;

// Tensor Backprop Training (simplified Adam hook)
// now using Embedding => model_dim independent of vocab_size
pub fn run(_opt: &str) {
    let pairs = load_pairs();
    let vocab = Vocab::build();
    let vocab_size = vocab.itos.len();

    let model_dim = 64;
    let mut encoder = EncoderT::new(6, vocab_size, model_dim, 1, 256);
    let mut decoder = DecoderT::new(6, vocab_size, model_dim, 1, 256);

    let lr = 0.001;
    let beam = 5;
    let start_id = *vocab.stoi.get(START).unwrap();
    let end_id = *vocab.stoi.get(END).unwrap();

    for epoch in 0..50 {
        for (src, tgt) in &pairs {
            // Encode
            let enc_x = to_matrix(src, vocab_size);
            let enc_out = encoder.forward(&enc_x);

            // Beam decoding
            let mut beams: Vec<(Vec<usize>, f32)> = vec![(vec![start_id], 0.0)];
            for _ in 0..50 {
                let mut new_beams = vec![];
                for (seq, sc) in &beams {
                    if *seq.last().unwrap() == end_id {
                        new_beams.push((seq.clone(), *sc));
                        continue;
                    }
                    let tin = to_matrix(&seq, vocab_size);
                    let logits = decoder.forward(
                        &Tensor::from_matrix(tin, true),
                        &enc_out,
                    );
                    let last = logits.data.rows - 1;
                    for tok in 0..vocab_size {
                        let p = logits.data.get(last, tok).exp();
                        let mut s = seq.clone();
                        s.push(tok);
                        new_beams.push((s, sc - p.ln()));
                    }
                }
                new_beams.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                beams = new_beams.into_iter().take(beam).collect();
            }
            let generated = &beams[0].0;

            // CrossEntropy-Loss
            let mut ce = 0.0;
            let mut cnt = 0.0;
            for (i, &tok) in tgt.iter().enumerate() {
                if i >= generated.len() {
                    break;
                }
                let t_in = to_matrix(&generated[..i + 1], vocab_size);
                let logits = decoder.forward(&Tensor::from_matrix(t_in, true), &enc_out);
                let last = logits.data.rows - 1;
                let mut sum = 0.0;
                for t in 0..vocab_size {
                    sum += logits.data.get(last, t).exp();
                }
                let p = logits.data.get(last, tok).exp() / sum;
                ce += -(p + 1e-9).ln();
                cnt += 1.0;
            }
            let loss = ce / cnt;
            println!("epoch {epoch} loss {loss}");

            // Adam-Update Placeholder
            for layer in &mut encoder.layers {
                for w in &mut layer.attn.wq.w.data.data {
                    *w -= lr * loss;
                }
            }
        }
    }

    // Save JSON async
    let model = json!({ "TODO": "export with embedding weights etc." });
    std::thread::spawn(move || {
        std::fs::write("model.json", serde_json::to_string(&model).unwrap()).unwrap();
        println!("model.json saved asynchronously");
    });
}
