use crate::data::{to_matrix, Vocab, START, END};
use crate::encoder_t::EncoderT;
use crate::decoder_t::DecoderT;
use crate::autograd::Tensor;
use crate::math::Matrix;
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
pub struct ModelJson {
    pub encoder_embedding: Vec<Vec<f32>>,
    pub decoder_embedding: Vec<Vec<f32>>,
}

pub fn load_model(path: &str, encoder: &mut EncoderT, decoder: &mut DecoderT) {
    let txt = fs::read_to_string(path).unwrap_or_else(|_| "{}".to_string());
    if let Ok(model) = serde_json::from_str::<ModelJson>(&txt) {
        let e_rows = model.encoder_embedding.len();
        let e_cols = model.encoder_embedding.get(0).map_or(0, |v| v.len());
        let e_flat: Vec<f32> = model.encoder_embedding.into_iter().flatten().collect();
        if e_rows > 0 && e_cols > 0 {
            let mat = Matrix::from_vec(e_rows, e_cols, e_flat);
            encoder.embedding.table.w = Tensor::from_matrix(mat, true);
        }
        let d_rows = model.decoder_embedding.len();
        let d_cols = model.decoder_embedding.get(0).map_or(0, |v| v.len());
        let d_flat: Vec<f32> = model.decoder_embedding.into_iter().flatten().collect();
        if d_rows > 0 && d_cols > 0 {
            let mat = Matrix::from_vec(d_rows, d_cols, d_flat);
            decoder.embedding.table.w = Tensor::from_matrix(mat, true);
        }
    }
    println!("Loaded weights from {}", path);
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
            let probs = Tensor::softmax(&logits);
            let last = probs.data.rows - 1;
            for tok in 0..vocab_size {
                let p = probs.data.get(last, tok);
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
