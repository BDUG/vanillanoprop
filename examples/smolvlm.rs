use std::error::Error;
use std::fs;

use vanillanoprop::config::Config;
use vanillanoprop::models::SmolVLM;
use vanillanoprop::weights;
use vanillanoprop::fetch_hf_files_with_cfg;

fn main() -> Result<(), Box<dyn Error>> {
    // Download configuration and weights for a tiny SmolVLM model.
    let cfg = Config::from_path("backprop_config.toml").unwrap_or_default();
    let files = fetch_hf_files_with_cfg("hf-internal-testing/tiny-random-smolvlm", &cfg)?;

    // Parse the configuration to determine model dimensions.
    let cfg_text = fs::read_to_string(&files.config)?;
    let cfg: serde_json::Value = serde_json::from_str(&cfg_text)?;

    let text_cfg = &cfg["text_config"];
    let vision_cfg = &cfg["vision_config"];

    let vocab_size = text_cfg["vocab_size"].as_u64().unwrap_or(1000) as usize;
    let text_dim = text_cfg["hidden_size"].as_u64().unwrap_or(32) as usize;
    let vision_dim = vision_cfg["hidden_size"].as_u64().unwrap_or(32) as usize;

    // Construct the model and attempt to load pretrained weights.
    let mut model = SmolVLM::new(vocab_size, vision_dim, text_dim);
    let _ = weights::load_smolvlm_from_hf(&files.config, &files.weights, &mut model);

    // Dummy 28x28 grayscale image and prompt tokens [0,1,2].
    let image = vec![0u8; 28 * 28];
    let prompt = [0usize, 1, 2];

    let fused = model.forward(&image, &prompt);
    println!("Fused embedding shape: {:?}", fused.shape);

    Ok(())
}
