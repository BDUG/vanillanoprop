use crate::autograd::Tensor;
use crate::math::Matrix;
use crate::models::{DecoderT, EncoderT};
use serde::{Deserialize, Serialize};
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
    let model = ModelJson {
        encoder_embedding: enc_emb,
        decoder_embedding: dec_emb,
    };
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
            let exp_rows = encoder.embedding.table.w.data.rows;
            let exp_cols = encoder.embedding.table.w.data.cols;
            let mut mat = Matrix::zeros(exp_rows, exp_cols);
            for r in 0..e_rows.min(exp_rows) {
                for c in 0..e_cols.min(exp_cols) {
                    mat.set(r, c, model.encoder_embedding[r][c]);
                }
            }
            encoder.embedding.table.w = Tensor::from_matrix(mat);
        }
        if !model.decoder_embedding.is_empty() {
            let d_rows = model.decoder_embedding.len();
            let d_cols = model.decoder_embedding[0].len();
            let exp_rows = decoder.embedding.table.w.data.rows;
            let exp_cols = decoder.embedding.table.w.data.cols;
            let mut mat = Matrix::zeros(exp_rows, exp_cols);
            for r in 0..d_rows.min(exp_rows) {
                for c in 0..d_cols.min(exp_cols) {
                    mat.set(r, c, model.decoder_embedding[r][c]);
                }
            }
            decoder.embedding.table.w = Tensor::from_matrix(mat);
        }
    }
    println!("Loaded weights from {}", path);
}
