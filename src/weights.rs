use crate::export::onnx::export_to_onnx;
use crate::layers::{Layer, LayerNorm, LinearT, MixtureOfExpertsT};
use crate::math::Matrix;
use crate::models::{
    DecoderT, EncoderT, LargeConceptModel, Sequential, SimpleCNN, TransformerEncoder, RNN, VAE,
};
use crate::tensor::Tensor;
use safetensors::tensor::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use std::path::Path;
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
    pub w3: Vec<Vec<f32>>,
    pub b3: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
pub struct RnnJson {
    pub params: Vec<Vec<Vec<f32>>>,
}

#[derive(Serialize, Deserialize)]
pub struct MoeJson {
    pub gate: Vec<Vec<f32>>,
    pub experts: Vec<Vec<Vec<f32>>>,
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
    assert!(t.shape.len() == 2, "tensor must be 2-D");
    let rows = t.shape[0];
    let cols = t.shape[1];
    (0..rows)
        .map(|r| (0..cols).map(|c| t.data[r * cols + c]).collect())
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

/// Quantize a slice of f32 weights into int8 values returning the quantized
/// bytes and scaling factor.
pub fn quantize(weights: &[f32]) -> (Vec<i8>, f32) {
    let max = weights.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let scale = if max == 0.0 { 1.0 } else { 127.0 / max };
    let q = weights
        .iter()
        .map(|&v| (v * scale).round().clamp(-128.0, 127.0) as i8)
        .collect();
    (q, scale)
}

/// Reconstruct floating point weights from quantized int8 data and scale.
pub fn dequantize(data: &[i8], scale: f32) -> Vec<f32> {
    let inv = if scale == 0.0 { 1.0 } else { 1.0 / scale };
    data.iter().map(|&v| v as f32 * inv).collect()
}

#[derive(Serialize, Deserialize)]
struct WeightsBin {
    weights: Vec<Vec<Vec<f32>>>,
}

/// Save a list of layer weights to a binary file using `bincode`.
///
/// The provided `params` should reference the `LinearT` layers whose weight
/// matrices will be serialised.  Only the raw weight matrices are persisted; any
/// optimiser state is ignored.
pub fn save_weights(path: &str, params: &[&LinearT]) -> Result<(), io::Error> {
    let weights: Vec<Vec<Vec<f32>>> = params.iter().map(|p| tensor_to_vec2(&p.w)).collect();
    let bin = bincode::serialize(&WeightsBin { weights })
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, bin)?;
    println!("Saved weights to {}", path);
    Ok(())
}

/// Load layer weights previously saved with [`save_weights`].
///
/// The loaded weights are copied into the provided `params` in order.  Any
/// mismatch in number of parameters or matrix shapes will simply truncate to
/// the smaller dimensions.
pub fn load_weights(path: &str, params: &mut [&mut LinearT]) -> Result<(), io::Error> {
    let bin = fs::read(path)?;
    let wb: WeightsBin =
        bincode::deserialize(&bin).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    for (p, w) in params.iter_mut().zip(wb.weights.iter()) {
        p.w = Tensor::from_matrix(vec2_to_matrix(w));
    }
    println!("Loaded weights from {}", path);
    Ok(())
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

pub fn save_moe(path: &str, moe: &mut MixtureOfExpertsT) -> Result<(), io::Error> {
    let gate = tensor_to_vec2(&moe.gate.w);
    let mut experts_w = Vec::new();
    for exp in moe.experts.iter_mut() {
        let params = exp.parameters();
        if let Some(p) = params.first() {
            experts_w.push(tensor_to_vec2(&p.w));
        }
    }
    let json = MoeJson {
        gate,
        experts: experts_w,
    };
    let txt = serde_json::to_string(&json).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    fs::write(path, txt)?;
    println!("Saved MoE weights to {}", path);
    Ok(())
}

pub fn load_moe(
    path: &str,
    input_dim: usize,
    output_dim: usize,
    num_experts: usize,
) -> Result<MixtureOfExpertsT, io::Error> {
    let txt = fs::read_to_string(path)?;
    let json: MoeJson =
        serde_json::from_str(&txt).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let experts: Vec<Box<dyn Layer>> = (0..num_experts)
        .map(|_| Box::new(LinearT::new(input_dim, output_dim)) as Box<dyn Layer>)
        .collect();
    let mut moe = MixtureOfExpertsT::new(input_dim, experts, 1);
    if !json.gate.is_empty() {
        moe.gate.w = Tensor::from_matrix(vec2_to_matrix(&json.gate));
    }
    for (exp, data) in moe.experts.iter_mut().zip(json.experts.iter()) {
        let mut params = exp.parameters();
        if let Some(p) = params.first_mut() {
            p.w = Tensor::from_matrix(vec2_to_matrix(data));
        }
    }
    println!("Loaded MoE weights from {}", path);
    Ok(moe)
}

/// Export a [`Sequential`] model to an ONNX file.
///
/// This is a thin wrapper around [`export_to_onnx`] and currently supports only
/// models constructed from layers that the exporter can map to ONNX operators.
pub fn save_onnx(path: &str, model: &Sequential) -> Result<(), Box<dyn std::error::Error>> {
    export_to_onnx(model, std::path::Path::new(path))
}

pub fn save_lcm(path: &str, model: &LargeConceptModel) -> Result<(), io::Error> {
    let (w1, b1, w2, b2, w3, b3) = model.parameters();
    let json = LcmJson {
        w1: matrix_to_vec2(w1),
        b1: b1.clone(),
        w2: matrix_to_vec2(w2),
        b2: b2.clone(),
        w3: matrix_to_vec2(w3),
        b3: b3.clone(),
    };
    let txt = serde_json::to_string(&json).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    fs::write(path, txt)?;
    println!("Saved LCM weights to {}", path);
    Ok(())
}

pub fn load_lcm(
    path: &str,
    input_dim: usize,
    hidden_dim1: usize,
    hidden_dim2: usize,
    num_classes: usize,
) -> Result<LargeConceptModel, io::Error> {
    let txt = fs::read_to_string(path)?;
    let json: LcmJson =
        serde_json::from_str(&txt).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let mut model = LargeConceptModel::new(input_dim, hidden_dim1, hidden_dim2, num_classes);
    let (w1, b1, w2, b2, w3, b3) = model.parameters_mut();
    if !json.w1.is_empty() {
        *w1 = vec2_to_matrix(&json.w1);
    }
    if !json.w2.is_empty() {
        *w2 = vec2_to_matrix(&json.w2);
    }
    if !json.w3.is_empty() {
        *w3 = vec2_to_matrix(&json.w3);
    }
    if !json.b1.is_empty() {
        *b1 = json.b1.clone();
    }
    if !json.b2.is_empty() {
        *b2 = json.b2.clone();
    }
    if !json.b3.is_empty() {
        *b3 = json.b3.clone();
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

/// Load a Transformer encoder from Hugging Face configuration and weights files.
///
/// `cfg_path` should point to a `config.json` file and `weights_path` to a
/// `model.safetensors` file. Only models following the BERT style naming
/// convention are currently supported.
pub fn load_transformer_from_hf(
    cfg_path: &Path,
    weights_path: &Path,
    model: &mut TransformerEncoder,
) -> Result<(), Box<dyn std::error::Error>> {
    #[derive(Deserialize)]
    struct HfConfig {
        num_hidden_layers: usize,
        hidden_size: usize,
        num_attention_heads: usize,
        intermediate_size: usize,
        vocab_size: usize,
    }

    let cfg_text = fs::read_to_string(cfg_path)?;
    let cfg: HfConfig = serde_json::from_str(&cfg_text)?;

    if cfg.num_hidden_layers != model.layers.len() {
        return Err(format!(
            "layer count mismatch: config {} vs model {}",
            cfg.num_hidden_layers,
            model.layers.len()
        )
        .into());
    }
    if cfg.hidden_size != model.embedding.table.w.data.cols {
        return Err(format!(
            "hidden size mismatch: config {} vs model {}",
            cfg.hidden_size, model.embedding.table.w.data.cols
        )
        .into());
    }
    if cfg.vocab_size != model.embedding.table.w.data.rows {
        return Err(format!(
            "vocab size mismatch: config {} vs model {}",
            cfg.vocab_size, model.embedding.table.w.data.rows
        )
        .into());
    }
    if cfg.num_attention_heads != model.layers[0].attn.num_heads {
        return Err(format!(
            "attention heads mismatch: config {} vs model {}",
            cfg.num_attention_heads, model.layers[0].attn.num_heads
        )
        .into());
    }
    if cfg.intermediate_size != model.layers[0].ff.w1.w.data.cols {
        return Err(format!(
            "feed-forward size mismatch: config {} vs model {}",
            cfg.intermediate_size, model.layers[0].ff.w1.w.data.cols
        )
        .into());
    }

    let weight_bytes = fs::read(weights_path)?;
    let tensors = SafeTensors::deserialize(&weight_bytes)?;

    // Embedding matrix
    load_embedding(
        &mut model.embedding.table,
        &tensors,
        "embeddings.word_embeddings.weight",
    )?;

    for i in 0..cfg.num_hidden_layers {
        let prefix = format!("encoder.layer.{}.", i);
        load_linear(
            &mut model.layers[i].attn.wq,
            &tensors,
            &(prefix.clone() + "attention.self.query.weight"),
        )?;
        load_linear(
            &mut model.layers[i].attn.wk,
            &tensors,
            &(prefix.clone() + "attention.self.key.weight"),
        )?;
        load_linear(
            &mut model.layers[i].attn.wv,
            &tensors,
            &(prefix.clone() + "attention.self.value.weight"),
        )?;
        load_linear(
            &mut model.layers[i].attn.wo,
            &tensors,
            &(prefix.clone() + "attention.output.dense.weight"),
        )?;
        load_layernorm(
            &mut model.layers[i].norm1,
            &tensors,
            &(prefix.clone() + "attention.output.LayerNorm.weight"),
            &(prefix.clone() + "attention.output.LayerNorm.bias"),
        )?;
        load_linear(
            &mut model.layers[i].ff.w1,
            &tensors,
            &(prefix.clone() + "intermediate.dense.weight"),
        )?;
        load_linear(
            &mut model.layers[i].ff.w2,
            &tensors,
            &(prefix.clone() + "output.dense.weight"),
        )?;
        load_layernorm(
            &mut model.layers[i].norm2,
            &tensors,
            &(prefix.clone() + "output.LayerNorm.weight"),
            &(prefix.clone() + "output.LayerNorm.bias"),
        )?;
    }

    Ok(())
}

fn to_f32_vec(t: &safetensors::tensor::TensorView) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if t.dtype() != Dtype::F32 {
        return Err("expected f32 tensor".into());
    }
    let mut out = Vec::with_capacity(t.data().len() / 4);
    for chunk in t.data().chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into()?));
    }
    Ok(out)
}

fn load_embedding(
    lin: &mut LinearT,
    tensors: &SafeTensors,
    name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let view = tensors
        .tensor(name)
        .map_err(|_| format!("missing tensor {name}"))?;
    let shape = view.shape();
    if shape.len() != 2 {
        return Err(format!("tensor {name} is not 2D").into());
    }
    let rows = shape[0];
    let cols = shape[1];
    if lin.w.data.rows != rows || lin.w.data.cols != cols {
        return Err(format!(
            "shape mismatch for {name}: expected {}x{}, got {}x{}",
            lin.w.data.rows, lin.w.data.cols, rows, cols
        )
        .into());
    }
    let data = to_f32_vec(&view)?;
    lin.w = Tensor::from_matrix(Matrix::from_vec(rows, cols, data));
    Ok(())
}

fn load_linear(
    lin: &mut LinearT,
    tensors: &SafeTensors,
    name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let view = tensors
        .tensor(name)
        .map_err(|_| format!("missing tensor {name}"))?;
    let shape = view.shape();
    if shape.len() != 2 {
        return Err(format!("tensor {name} is not 2D").into());
    }
    let out_dim = shape[0];
    let in_dim = shape[1];
    if lin.w.data.rows != in_dim || lin.w.data.cols != out_dim {
        return Err(format!(
            "shape mismatch for {name}: expected {}x{}, got {}x{}",
            lin.w.data.rows, lin.w.data.cols, in_dim, out_dim
        )
        .into());
    }
    let data = to_f32_vec(&view)?;
    let mut mat = Matrix::zeros(in_dim, out_dim);
    for r in 0..out_dim {
        for c in 0..in_dim {
            mat.set(c, r, data[r * in_dim + c]);
        }
    }
    lin.w = Tensor::from_matrix(mat);
    Ok(())
}

fn load_layernorm(
    norm: &mut LayerNorm,
    tensors: &SafeTensors,
    w_name: &str,
    b_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let w_view = tensors
        .tensor(w_name)
        .map_err(|_| format!("missing tensor {w_name}"))?;
    let b_view = tensors
        .tensor(b_name)
        .map_err(|_| format!("missing tensor {b_name}"))?;
    let w = to_f32_vec(&w_view)?;
    let b = to_f32_vec(&b_view)?;
    if norm.gamma.w.len() != w.len() || norm.beta.w.len() != b.len() {
        return Err(format!("LayerNorm shape mismatch for {w_name}").into());
    }
    norm.gamma.w.copy_from_slice(&w);
    norm.beta.w.copy_from_slice(&b);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype, TensorView};

    #[test]
    fn to_f32_vec_errors_on_non_f32() {
        let data = vec![0u8; 4];
        let view =
            TensorView::new(Dtype::U8, vec![4], &data).expect("failed to create tensor view");
        let err = to_f32_vec(&view).expect_err("expected error for non-f32 tensor");
        assert!(err.to_string().contains("expected f32 tensor"));
    }
}
