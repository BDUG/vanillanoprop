use std::error::Error;
use std::fs;

use vanillanoprop::huggingface;
use vanillanoprop::math::Matrix;
use vanillanoprop::models::{ResNet, TransformerEncoder};
use vanillanoprop::weights;

fn main() -> Result<(), Box<dyn Error>> {
    // Download configuration and weights for a tiny CLIP-like model.
    let files = huggingface::fetch_hf_files("hf-internal-testing/tiny-random-clip", None)?;

    // Read the top-level configuration file and extract nested text and vision configs.
    let cfg_text = fs::read_to_string(&files.config)?;
    let cfg: serde_json::Value = serde_json::from_str(&cfg_text)?;

    let text_cfg = &cfg["text_config"];
    let txt_layers = text_cfg["num_hidden_layers"].as_u64().unwrap_or(1) as usize;
    let txt_hidden = text_cfg["hidden_size"].as_u64().unwrap_or(32) as usize;
    let txt_heads = text_cfg["num_attention_heads"].as_u64().unwrap_or(4) as usize;
    let txt_inter = text_cfg["intermediate_size"].as_u64().unwrap_or(64) as usize;
    let vocab_size = text_cfg["vocab_size"].as_u64().unwrap_or(1000) as usize;

    // Build the text transformer using parameters from the configuration.
    let mut text_enc = TransformerEncoder::new(
        txt_layers, vocab_size, txt_hidden, txt_heads, txt_inter, 0.0,
    );

    // Attempt to load pretrained weights for the text encoder.
    let _ = weights::load_transformer_from_hf(&files.config, &files.weights, &mut text_enc);

    // Vision configuration for a ResNet-based image encoder.
    let vision_cfg = &cfg["vision_config"];
    let vis_layers = vision_cfg["num_hidden_layers"].as_u64().unwrap_or(1) as usize;
    let vis_hidden = vision_cfg["hidden_size"].as_u64().unwrap_or(32) as usize;

    // Reuse the existing ResNet implementation to obtain image embeddings.
    let vision = ResNet::new(1, vis_hidden, vis_layers);

    // Dummy 28x28 grayscale image (all zeros) and prompt tokens [0,1,2].
    let image = vec![0u8; 28 * 28];
    let (img_feat, _) = vision.forward(&image);

    let tokens = [0usize, 1, 2];
    let mut x = Matrix::zeros(tokens.len(), vocab_size);
    for (t, &id) in tokens.iter().enumerate() {
        x.set(t, id, 1.0);
    }
    let txt_feat = text_enc.forward(x, None);

    println!(
        "Embeddings -> image: {} dims, text: {}x{}",
        img_feat.len(),
        txt_feat.shape[0],
        txt_feat.shape[1]
    );

    Ok(())
}
