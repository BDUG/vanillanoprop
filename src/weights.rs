use crate::encoder_t::EncoderT;
use crate::decoder_t::DecoderT;
use crate::autograd::Tensor;
use crate::math::Matrix;
use serde::{Serialize, Deserialize};
use std::fs;

#[derive(Serialize, Deserialize)]
pub struct ModelJson {
    pub encoder_embedding: Vec<Vec<f32>>,
    pub decoder_embedding: Vec<Vec<f32>>,
}

fn tensor_to_vec2(t: &Tensor) -> Vec<Vec<f32>> {
    let m = &t.data;
    (0..m.rows)
        .map(|r| (0..m.cols).map(|c| m.get(r, c)).collect())
        .collect()
}

pub fn save_model(path: &str, encoder: &EncoderT, decoder: Option<&DecoderT>) {
    let enc_emb = tensor_to_vec2(&encoder.embedding.table.w);
    let dec_emb = decoder
        .map(|d| tensor_to_vec2(&d.embedding.table.w))
        .unwrap_or_default();
    let model = ModelJson { encoder_embedding: enc_emb, decoder_embedding: dec_emb };
    if let Ok(txt) = serde_json::to_string(&model) {
        let _ = fs::write(path, txt);
        println!("Saved weights to {}", path);
    }
}

pub fn load_model(path: &str, encoder: &mut EncoderT, decoder: &mut DecoderT) {
    let txt = fs::read_to_string(path).unwrap_or_else(|_| "{}".to_string());
    if let Ok(model) = serde_json::from_str::<ModelJson>(&txt) {
        if !model.encoder_embedding.is_empty() {
            let e_rows = model.encoder_embedding.len();
            let e_cols = model.encoder_embedding[0].len();
            let e_flat: Vec<f32> = model.encoder_embedding.into_iter().flatten().collect();
            let mat = Matrix::from_vec(e_rows, e_cols, e_flat);
            encoder.embedding.table.w = Tensor::from_matrix(mat);
        }
        if !model.decoder_embedding.is_empty() {
            let d_rows = model.decoder_embedding.len();
            let d_cols = model.decoder_embedding[0].len();
            let d_flat: Vec<f32> = model.decoder_embedding.into_iter().flatten().collect();
            let mat = Matrix::from_vec(d_rows, d_cols, d_flat);
            decoder.embedding.table.w = Tensor::from_matrix(mat);
        }
    }
    println!("Loaded weights from {}", path);
}

