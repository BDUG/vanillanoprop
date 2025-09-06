use serde::Deserialize;
use std::fs;
use vanillanoprop::layers::Activation;
use vanillanoprop::models::{DecoderT, TransformerEncoder};
use vanillanoprop::reward::NGramReward;
use vanillanoprop::rl::{LanguageEnv, SelfAdaptAgent};

mod common;

#[derive(Deserialize)]
struct SelfAdaptConfig {
    reference: String,
    episodes: usize,
    lr: f32,
    #[serde(default = "default_model_dim")]
    model_dim: usize,
}

fn default_model_dim() -> usize {
    32
}

impl SelfAdaptConfig {
    fn from_path(path: &str) -> Option<Self> {
        let content = fs::read_to_string(path).ok()?;
        toml::from_str(&content).ok()
    }
}

fn main() {
    let args = common::init_logging();
    let mut config_path = "configs/self_adapt_config.toml".to_string();
    let mut lr_arg = None;
    let mut episodes_arg = None;
    let mut reference_arg = None;
    let mut model_dim_arg = None;

    let mut i = 1; // skip bin name
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                if i + 1 < args.len() {
                    config_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--lr" => {
                if i + 1 < args.len() {
                    lr_arg = args[i + 1].parse().ok();
                    i += 1;
                }
            }
            "--episodes" => {
                if i + 1 < args.len() {
                    episodes_arg = args[i + 1].parse().ok();
                    i += 1;
                }
            }
            "--reference" => {
                if i + 1 < args.len() {
                    reference_arg = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--model-dim" => {
                if i + 1 < args.len() {
                    model_dim_arg = args[i + 1].parse().ok();
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let mut cfg = SelfAdaptConfig::from_path(&config_path)
        .unwrap_or_else(|| panic!("failed to load config {config_path}"));
    if let Some(lr) = lr_arg {
        cfg.lr = lr;
    }
    if let Some(ep) = episodes_arg {
        cfg.episodes = ep;
    }
    if let Some(r) = reference_arg {
        cfg.reference = r;
    }
    if let Some(md) = model_dim_arg {
        cfg.model_dim = md;
    }

    let vocab_size = 256;
    let env = LanguageEnv::new(cfg.reference.as_bytes().to_vec());
    let encoder = TransformerEncoder::new(1, vocab_size, cfg.model_dim, 2, cfg.model_dim * 2, 0.0);
    let decoder = DecoderT::new(
        1,
        vocab_size,
        cfg.model_dim,
        cfg.model_dim * 2,
        Activation::ReLU,
        false,
        1,
    );
    let reward = NGramReward::new(1);
    let mut agent = SelfAdaptAgent::new(env, encoder, decoder, cfg.lr, vocab_size, reward);

    for ep in 0..cfg.episodes {
        agent.reset();
        let mut total = 0.0f32;
        while let Some(r) = agent.step() {
            total += r;
        }
        log::info!("episode {} reward {total:.2}", ep + 1);
    }
}
