use hf_hub::api::sync::ApiBuilder;
use std::error::Error;
use std::path::{Path, PathBuf};

/// Local paths to important files fetched from the Hugging Face Hub.
pub struct HfFiles {
    pub config: PathBuf,
    pub weights: PathBuf,
    pub tokenizer: Option<PathBuf>,
}

/// Download `config.json` and weights for `model_id` from the Hugging Face Hub.
///
/// If `cache_dir` is provided, the files will be cached under this directory,
/// otherwise the default cache location of `hf-hub` is used.
pub fn fetch_hf_files(model_id: &str, cache_dir: Option<&Path>) -> Result<HfFiles, Box<dyn Error>> {
    let builder = if let Some(dir) = cache_dir {
        ApiBuilder::new().with_cache_dir(dir.to_path_buf())
    } else {
        ApiBuilder::new()
    };
    let api = builder.build()?;
    let repo = api.model(model_id.to_string());
    let config = repo.get("config.json")?;
    let weights = match repo.get("model.safetensors") {
        Ok(p) => p,
        Err(_) => repo.get("pytorch_model.bin")?,
    };
    let tokenizer = match repo.get("tokenizer.model") {
        Ok(p) => Some(p),
        Err(_) => None,
    };
    Ok(HfFiles {
        config,
        weights,
        tokenizer,
    })
}
