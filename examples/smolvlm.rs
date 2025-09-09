// Compile with `--features vlm` to enable the `image` crate and internal tokenizer.
// The program prints the model's textual answer to the supplied question.
#[cfg(feature = "vlm")]
use std::env;
#[cfg(feature = "vlm")]
use std::error::Error;
#[cfg(feature = "vlm")]
use std::fs;
#[cfg(feature = "vlm")]
use std::process;

#[cfg(feature = "vlm")]
use image::imageops::FilterType;
#[cfg(feature = "vlm")]
use vanillanoprop::config::Config;
#[cfg(feature = "vlm")]
use vanillanoprop::fetch_hf_files_with_cfg;
#[cfg(feature = "vlm")]
use vanillanoprop::math::Matrix;
#[cfg(feature = "vlm")]
use vanillanoprop::models::SmolVLM;
#[cfg(feature = "vlm")]
use vanillanoprop::tokenizer::Tokenizer;
#[cfg(feature = "vlm")]
use vanillanoprop::weights;

#[cfg(feature = "vlm")]
fn matrix_to_ids(m: &Matrix) -> Vec<u32> {
    let mut ids = Vec::with_capacity(m.rows);
    for r in 0..m.rows {
        let mut best = 0;
        let mut best_val = f32::MIN;
        for c in 0..m.cols {
            let v = m.get(r, c);
            if v > best_val {
                best_val = v;
                best = c;
            }
        }
        ids.push(best as u32);
    }
    ids
}

#[cfg(feature = "vlm")]
fn main() -> Result<(), Box<dyn Error>> {
    // Expect an image path as the first argument and an optional question.
    let mut args = env::args().skip(1);
    let path = args.next().unwrap_or_else(|| {
        eprintln!("Usage: cargo run --example smolvlm --features vlm <IMAGE_PATH> <QUESTION>");
        process::exit(1);
    });
    let question = args
        .next()
        .unwrap_or_else(|| "Describe the image".to_string());

    // Download configuration and weights for a tiny SmolVLM model.
    let cfg = Config::from_path("configs/smolvlm.toml").unwrap_or_default();
    let files = fetch_hf_files_with_cfg("katuni4ka/tiny-random-smolvlm2", &cfg)?;

    // Load tokenizer for mapping ids back to text using the internal implementation.
    let tok_path = files.tokenizer_json.ok_or("tokenizer.json not found")?;
    let tokenizer = Tokenizer::from_json(&tok_path)?;

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

    // Load the image, convert to 28x28 grayscale, and collect bytes.
    let image = image::open(path)?
        .grayscale()
        .resize_exact(28, 28, FilterType::Triangle)
        .to_luma8()
        .into_raw();
    let prompt: Vec<usize> = tokenizer
        .encode(&question)
        .into_iter()
        .map(|id| id as usize)
        .collect();

    let fused = model.forward(&image, &prompt);
    #[cfg(debug_assertions)]
    {
        // Display the full fused embedding tensor for debugging.
        println!("Fused embedding: {:?}", fused);
    }

    let fused_m = Matrix {
        rows: fused.shape[0],
        cols: fused.shape[1],
        data: fused.data.clone(),
    };
    let ids = matrix_to_ids(&fused_m);
    #[cfg(debug_assertions)]
    {
        println!("Token IDs: {:?}", ids);
    }
    let text = tokenizer.decode(&ids);
    println!("Answer: {}", text);

    Ok(())
}

#[cfg(not(feature = "vlm"))]
fn main() {
    eprintln!(
        "This example requires the `vlm` feature. Run with `cargo run --example smolvlm --features vlm <IMAGE_PATH> <QUESTION>`."
    );
}
