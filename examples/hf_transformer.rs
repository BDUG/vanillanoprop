use std::error::Error;
use std::fs;

use vanillanoprop::huggingface;
use vanillanoprop::math::Matrix;
use vanillanoprop::models::TransformerEncoder;
use vanillanoprop::weights;

fn main() -> Result<(), Box<dyn Error>> {
    // Download configuration and weights for a tiny BERT model.
    let files = huggingface::fetch_hf_files("hf-internal-testing/tiny-random-bert", None)?;

    // Read dimensions from the Hugging Face configuration file.
    #[derive(serde::Deserialize)]
    struct HfConfig {
        num_hidden_layers: usize,
        hidden_size: usize,
        num_attention_heads: usize,
        intermediate_size: usize,
        vocab_size: usize,
    }

    let cfg_text = fs::read_to_string(&files.config)?;
    let cfg: HfConfig = serde_json::from_str(&cfg_text)?;

    // Build a transformer using the sizes from the configuration.
    let mut enc = TransformerEncoder::new(
        cfg.num_hidden_layers,
        cfg.vocab_size,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.intermediate_size,
        0.0,
    );

    // Load pretrained weights.
    weights::load_transformer_from_hf(&files.config, &files.weights, &mut enc)?;

    // Run a dummy inference with token ids [0, 1, 2].
    let tokens = [0usize, 1, 2];
    let mut x = Matrix::zeros(tokens.len(), cfg.vocab_size);
    for (t, &id) in tokens.iter().enumerate() {
        x.set(t, id, 1.0);
    }

    let h = enc.forward(x, None);
    println!("Output shape: {}x{}", h.shape[0], h.shape[1]);

    Ok(())
}
