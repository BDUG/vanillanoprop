use hf_hub::api::sync::{ApiBuilder, ApiError};
use std::error::Error;
use std::path::{Path, PathBuf};

use crate::config::Config;

/// Local paths to important files fetched from the Hugging Face Hub.
pub struct HfFiles {
    pub config: PathBuf,
    pub weights: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub tokenizer_json: Option<PathBuf>,
    pub processor: Option<PathBuf>,
}

/// Download `config.json` and weights for `model_id` from the Hugging Face Hub.
///
/// If `cache_dir` is provided, the files will be cached under this directory,
/// otherwise the default cache location of `hf-hub` is used.
pub fn fetch_hf_files(
    model_id: &str,
    cache_dir: Option<&Path>,
    token: Option<&str>,
) -> Result<HfFiles, Box<dyn Error>> {
    let mut builder = if let Some(dir) = cache_dir {
        ApiBuilder::new().with_cache_dir(dir.to_path_buf())
    } else {
        ApiBuilder::new()
    };
    if let Some(t) = token {
        builder = builder.with_token(Some(t.to_string()));
    }
    let api = builder.build()?;
    let repo = api.model(model_id.to_string());
    let map_err = |e: ApiError| -> Box<dyn Error> {
        if let ApiError::RequestError(err) = &e {
            if let ureq::Error::Status(401, _) = **err {
                return "Invalid or expired Hugging Face token".into();
            }
        }
        e.into()
    };
    let config = repo.get("config.json").map_err(map_err)?;
    let weights = match repo.get("model.safetensors") {
        Ok(p) => p,
        Err(e) => {
            if let ApiError::RequestError(err) = &e {
                if let ureq::Error::Status(401, _) = **err {
                    return Err("Invalid or expired Hugging Face token".into());
                }
            }
            repo.get("pytorch_model.bin").map_err(map_err)?
        }
    };
    let tokenizer = repo.get("tokenizer.model").ok();
    let tokenizer_json = repo.get("tokenizer.json").ok();
    let processor = repo
        .get("preprocessor_config.json")
        .or_else(|_| repo.get("image_processor.json"))
        .ok();
    Ok(HfFiles {
        config,
        weights,
        tokenizer,
        tokenizer_json,
        processor,
    })
}

/// Convenience wrapper to fetch files using a [`Config`] for authentication.
///
/// Uses the `hf_token` field from the provided configuration if present.
pub fn fetch_hf_files_with_cfg(
    model_id: &str,
    cfg: &Config,
) -> Result<HfFiles, Box<dyn Error>> {
    fetch_hf_files(model_id, None, cfg.hf_token.as_deref())
}
