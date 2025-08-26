use crate::tensor::Tensor;
use crate::math::Matrix;
use crate::models::{DecoderT, EncoderT, SimpleCNN};
use serde::{Deserialize, Serialize};
use std::{fs, io};

#[derive(Serialize, Deserialize)]
pub struct ModelJson {
    pub encoder_embedding: Vec<Vec<f32>>,
    pub decoder_embedding: Vec<Vec<f32>>,
}

#[derive(Serialize, Deserialize)]
pub struct CnnJson {
    pub fc: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}

fn tensor_to_vec2(t: &Tensor) -> Vec<Vec<f32>> {
    let m = &t.data;
    (0..m.rows)
        .map(|r| (0..m.cols).map(|c| m.get(r, c)).collect())
        .collect()
}

pub fn save_model(
    path: &str,
    encoder: &mut EncoderT,
    decoder: Option<&mut DecoderT>,
) -> Result<(), io::Error> {
    let enc_emb = {
        let params = encoder.embedding.parameters();
        tensor_to_vec2(&params[0].w)
    };
    let dec_emb = if let Some(d) = decoder {
        let params = d.embedding.parameters();
        tensor_to_vec2(&params[0].w)
    } else {
        Vec::new()
    };
    let model = ModelJson {
        encoder_embedding: enc_emb,
        decoder_embedding: dec_emb,
    };
    let txt = serde_json::to_string(&model)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    fs::write(path, txt)?;
    println!("Saved weights to {}", path);
    Ok(())
}

pub fn load_model(
    path: &str,
    encoder: &mut EncoderT,
    decoder: &mut DecoderT,
) -> Result<(), io::Error> {
    let txt = fs::read_to_string(path)?;
    let model: ModelJson = serde_json::from_str(&txt)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    if !model.encoder_embedding.is_empty() {
        let e_rows = model.encoder_embedding.len();
        let e_cols = model.encoder_embedding[0].len();
        let mut params = encoder.embedding.parameters();
        let exp_rows = params[0].w.data.rows;
        let exp_cols = params[0].w.data.cols;
        let mut mat = Matrix::zeros(exp_rows, exp_cols);
        for r in 0..e_rows.min(exp_rows) {
            for c in 0..e_cols.min(exp_cols) {
                mat.set(r, c, model.encoder_embedding[r][c]);
            }
        }
        params[0].w = Tensor::from_matrix(mat);
    }
    if !model.decoder_embedding.is_empty() {
        let d_rows = model.decoder_embedding.len();
        let d_cols = model.decoder_embedding[0].len();
        let mut params = decoder.embedding.parameters();
        let exp_rows = params[0].w.data.rows;
        let exp_cols = params[0].w.data.cols;
        let mut mat = Matrix::zeros(exp_rows, exp_cols);
        for r in 0..d_rows.min(exp_rows) {
            for c in 0..d_cols.min(exp_cols) {
                mat.set(r, c, model.decoder_embedding[r][c]);
            }
        }
        params[0].w = Tensor::from_matrix(mat);
    }
    println!("Loaded weights from {}", path);
    Ok(())
}

fn matrix_to_vec2(m: &Matrix) -> Vec<Vec<f32>> {
    (0..m.rows)
        .map(|r| (0..m.cols).map(|c| m.get(r, c)).collect())
        .collect()
}

pub fn save_cnn(path: &str, cnn: &SimpleCNN) -> Result<(), io::Error> {
    let (fc, bias) = cnn.parameters();
    let model = CnnJson {
        fc: matrix_to_vec2(fc),
        bias: bias.clone(),
    };
    let txt = serde_json::to_string(&model)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    fs::write(path, txt)?;
    println!("Saved CNN weights to {}", path);
    Ok(())
}

pub fn load_cnn(path: &str, num_classes: usize) -> Result<SimpleCNN, io::Error> {
    let txt = fs::read_to_string(path)?;
    let model: CnnJson = serde_json::from_str(&txt)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let mut cnn = SimpleCNN::new(num_classes);
    let (fc, bias) = cnn.parameters_mut();
    if !model.fc.is_empty() {
        let rows = model.fc.len();
        let cols = model.fc[0].len();
        let mut mat = Matrix::zeros(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                mat.set(r, c, model.fc[r][c]);
            }
        }
        *fc = mat;
    }
    if !model.bias.is_empty() {
        *bias = model.bias;
    }
    println!("Loaded CNN weights from {}", path);
    Ok(cnn)
}
