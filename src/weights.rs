use crate::math::Matrix;
use crate::models::{DecoderT, EncoderT, LargeConceptModel, SimpleCNN, RNN, VAE};
use crate::tensor::Tensor;
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

#[derive(Serialize, Deserialize)]
pub struct LcmJson {
    pub w1: Vec<Vec<f32>>,
    pub b1: Vec<f32>,
    pub w2: Vec<Vec<f32>>,
    pub b2: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
pub struct RnnJson {
    pub params: Vec<Vec<Vec<f32>>>,
}

#[derive(Serialize, Deserialize)]
pub struct VaeJson {
    pub enc_fc1: Vec<Vec<f32>>,
    pub enc_mu: Vec<Vec<f32>>,
    pub enc_logvar: Vec<Vec<f32>>,
    pub dec_fc1: Vec<Vec<f32>>,
    pub dec_fc2: Vec<Vec<f32>>,
}

/// Convert a [`Tensor`] into a 2-D `Vec` for serialisation.
pub fn tensor_to_vec2(t: &Tensor) -> Vec<Vec<f32>> {
    let m = &t.data;
    (0..m.rows)
        .map(|r| (0..m.cols).map(|c| m.get(r, c)).collect())
        .collect()
}

/// Convert a 2-D `Vec` into a [`Matrix`].
pub fn vec2_to_matrix(rows: &[Vec<f32>]) -> Matrix {
    if rows.is_empty() || rows[0].is_empty() {
        return Matrix::zeros(0, 0);
    }
    let r = rows.len();
    let c = rows[0].len();
    let mut mat = Matrix::zeros(r, c);
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            mat.set(i, j, val);
        }
    }
    mat
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
    let txt = serde_json::to_string(&model).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
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
    let model: ModelJson =
        serde_json::from_str(&txt).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    if !model.encoder_embedding.is_empty() {
        let mut params = encoder.embedding.parameters();
        let exp_rows = params[0].w.data.rows;
        let exp_cols = params[0].w.data.cols;
        let loaded = vec2_to_matrix(&model.encoder_embedding);
        let mut mat = Matrix::zeros(exp_rows, exp_cols);
        for r in 0..loaded.rows.min(exp_rows) {
            for c in 0..loaded.cols.min(exp_cols) {
                mat.set(r, c, loaded.get(r, c));
            }
        }
        params[0].w = Tensor::from_matrix(mat);
    }
    if !model.decoder_embedding.is_empty() {
        let mut params = decoder.embedding.parameters();
        let exp_rows = params[0].w.data.rows;
        let exp_cols = params[0].w.data.cols;
        let loaded = vec2_to_matrix(&model.decoder_embedding);
        let mut mat = Matrix::zeros(exp_rows, exp_cols);
        for r in 0..loaded.rows.min(exp_rows) {
            for c in 0..loaded.cols.min(exp_cols) {
                mat.set(r, c, loaded.get(r, c));
            }
        }
        params[0].w = Tensor::from_matrix(mat);
    }
    println!("Loaded weights from {}", path);
    Ok(())
}

/// Convert a [`Matrix`] into a 2-D `Vec` for serialisation.
pub fn matrix_to_vec2(m: &Matrix) -> Vec<Vec<f32>> {
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
    let txt = serde_json::to_string(&model).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    fs::write(path, txt)?;
    println!("Saved CNN weights to {}", path);
    Ok(())
}

pub fn load_cnn(path: &str, num_classes: usize) -> Result<SimpleCNN, io::Error> {
    let txt = fs::read_to_string(path)?;
    let model: CnnJson =
        serde_json::from_str(&txt).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let mut cnn = SimpleCNN::new(num_classes);
    let (fc, bias) = cnn.parameters_mut();
    if !model.fc.is_empty() {
        *fc = vec2_to_matrix(&model.fc);
    }
    if !model.bias.is_empty() {
        *bias = model.bias;
    }
    println!("Loaded CNN weights from {}", path);
    Ok(cnn)
}

pub fn save_lcm(path: &str, model: &LargeConceptModel) -> Result<(), io::Error> {
    let (w1, b1, w2, b2) = model.parameters();
    let json = LcmJson {
        w1: matrix_to_vec2(w1),
        b1: b1.clone(),
        w2: matrix_to_vec2(w2),
        b2: b2.clone(),
    };
    let txt = serde_json::to_string(&json).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    fs::write(path, txt)?;
    println!("Saved LCM weights to {}", path);
    Ok(())
}

pub fn load_lcm(
    path: &str,
    input_dim: usize,
    hidden_dim: usize,
    num_classes: usize,
) -> Result<LargeConceptModel, io::Error> {
    let txt = fs::read_to_string(path)?;
    let json: LcmJson =
        serde_json::from_str(&txt).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let mut model = LargeConceptModel::new(input_dim, hidden_dim, num_classes);
    let (w1, b1, w2, b2) = model.parameters_mut();
    if !json.w1.is_empty() {
        *w1 = vec2_to_matrix(&json.w1);
    }
    if !json.w2.is_empty() {
        *w2 = vec2_to_matrix(&json.w2);
    }
    if !json.b1.is_empty() {
        *b1 = json.b1.clone();
    }
    if !json.b2.is_empty() {
        *b2 = json.b2.clone();
    }
    println!("Loaded LCM weights from {}", path);
    Ok(model)
}

pub fn save_rnn(path: &str, model: &mut RNN) -> Result<(), io::Error> {
    let mut params = Vec::new();
    for p in model.parameters() {
        params.push(tensor_to_vec2(&p.w));
    }
    let json = RnnJson { params };
    let txt = serde_json::to_string(&json).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    fs::write(path, txt)?;
    println!("Saved RNN weights to {}", path);
    Ok(())
}

pub fn load_rnn(
    path: &str,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
) -> Result<RNN, io::Error> {
    let txt = fs::read_to_string(path)?;
    let json: RnnJson =
        serde_json::from_str(&txt).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let mut model = RNN::new_gru(input_dim, hidden_dim, output_dim);
    let mut params = model.parameters();
    for (p, data) in params.iter_mut().zip(json.params.iter()) {
        p.w = Tensor::from_matrix(vec2_to_matrix(data));
    }
    println!("Loaded RNN weights from {}", path);
    Ok(model)
}

/// Save an arbitrary checkpoint structure to `path` using JSON serialisation.
pub fn save_checkpoint<T: Serialize>(path: &str, state: &T) -> Result<(), io::Error> {
    let txt = serde_json::to_string(state).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, txt)?;
    println!("Saved checkpoint to {}", path);
    Ok(())
}

/// Load a checkpoint from `path` that was saved with [`save_checkpoint`].
pub fn load_checkpoint<T: for<'de> Deserialize<'de>>(path: &str) -> Result<T, io::Error> {
    let txt = fs::read_to_string(path)?;
    let state = serde_json::from_str(&txt).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    println!("Loaded checkpoint from {}", path);
    Ok(state)
}

pub fn save_vae(path: &str, vae: &VAE) -> Result<(), io::Error> {
    let json = VaeJson {
        enc_fc1: tensor_to_vec2(&vae.enc_fc1.w),
        enc_mu: tensor_to_vec2(&vae.enc_mu.w),
        enc_logvar: tensor_to_vec2(&vae.enc_logvar.w),
        dec_fc1: tensor_to_vec2(&vae.dec_fc1.w),
        dec_fc2: tensor_to_vec2(&vae.dec_fc2.w),
    };
    let txt = serde_json::to_string(&json).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    fs::write(path, txt)?;
    println!("Saved VAE weights to {}", path);
    Ok(())
}

pub fn load_vae(
    path: &str,
    input_dim: usize,
    hidden_dim: usize,
    latent_dim: usize,
) -> Result<VAE, io::Error> {
    let txt = fs::read_to_string(path)?;
    let json: VaeJson =
        serde_json::from_str(&txt).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let mut vae = VAE::new(input_dim, hidden_dim, latent_dim);
    if !json.enc_fc1.is_empty() {
        vae.enc_fc1.w = Tensor::from_matrix(vec2_to_matrix(&json.enc_fc1));
    }
    if !json.enc_mu.is_empty() {
        vae.enc_mu.w = Tensor::from_matrix(vec2_to_matrix(&json.enc_mu));
    }
    if !json.enc_logvar.is_empty() {
        vae.enc_logvar.w = Tensor::from_matrix(vec2_to_matrix(&json.enc_logvar));
    }
    if !json.dec_fc1.is_empty() {
        vae.dec_fc1.w = Tensor::from_matrix(vec2_to_matrix(&json.dec_fc1));
    }
    if !json.dec_fc2.is_empty() {
        vae.dec_fc2.w = Tensor::from_matrix(vec2_to_matrix(&json.dec_fc2));
    }
    println!("Loaded VAE weights from {}", path);
    Ok(vae)
}
