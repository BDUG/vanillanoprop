use crate::data::{to_matrix, Vocab, START, END};
use crate::encoder_t::EncoderT;
use crate::decoder_t::DecoderT;
use crate::autograd::Tensor;
use serde::{Deserialize};
use serde_json::Value;
use std::fs;

#[derive(Deserialize)]
pub struct ModelJson {
    pub encoder: Vec<Value>,
    pub decoder: Vec<Value>,
}

pub fn load_model(path: &str, _encoder: &mut EncoderT, _decoder: &mut DecoderT) {
    let txt = fs::read_to_string(path).unwrap();
    let _model: ModelJson = serde_json::from_str(&txt).unwrap();
    println!("Loaded weights from {}", path);

    // TODO: map into embedding + transformer weights (currently dummy)
}

pub fn run(input: &str) {
    let vocab = Vocab::build();
    let vocab_size = vocab.itos.len();
    let start_id = *vocab.stoi.get(START).unwrap();
    let end_id = *vocab.stoi.get(END).unwrap();

    let model_dim = 64;
    let mut encoder = EncoderT::new(6, vocab_size, model_dim, 1, 256);
    let mut decoder = DecoderT::new(6, vocab_size, model_dim, 1, 256);

    load_model("model.json", &mut encoder, &mut decoder);

    let src = vocab.encode(input);
    let enc_x = to_matrix(&src, vocab_size);
    let enc_out = encoder.forward(&enc_x);

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
        beams = new_beams.into_iter().take(5).collect();
    }
    let out_ids = &beams[0].0;
    let translation = vocab.decode(out_ids);
    println!("{{\"input\":\"{}\", \"translation\":\"{}\"}}", input, translation);
}
