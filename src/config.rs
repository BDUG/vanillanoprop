use serde::Deserialize;
use std::fs;

/// Training configuration loaded from a TOML or JSON file.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Number of training epochs.
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    /// Batch size used when loading data.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Name of dataset to use for training.
    #[serde(default = "default_dataset")]
    pub dataset: String,
    /// Discount factor for reinforcement learning agents.
    #[serde(default = "default_gamma")]
    pub gamma: f32,
    /// Lambda parameter used by certain advantage estimators.
    #[serde(default = "default_lam")]
    pub lam: f32,
    /// Maximum search depth for tree‑based agents.
    #[serde(default = "default_max_depth")]
    pub max_depth: usize,
    /// Number of rollout steps taken during planning.
    #[serde(default = "default_rollout_steps")]
    pub rollout_steps: usize,
    /// Candidate learning rates for AutoML search. When multiple values are
    /// provided the [`automl`](crate::automl) module will explore them.
    #[serde(default = "default_learning_rates")]
    pub learning_rate: Vec<f32>,
    /// Whether to enable quantized inference where supported.
    #[serde(default)]
    pub quantized: bool,
}

fn default_epochs() -> usize {
    5
}

fn default_batch_size() -> usize {
    4
}

fn default_dataset() -> String {
    "mnist".into()
}

fn default_gamma() -> f32 {
    0.99
}

fn default_lam() -> f32 {
    0.95
}

fn default_max_depth() -> usize {
    10
}

fn default_rollout_steps() -> usize {
    10
}

fn default_learning_rates() -> Vec<f32> {
    vec![0.001]
}

impl Default for Config {
    fn default() -> Self {
        Self {
            epochs: default_epochs(),
            batch_size: default_batch_size(),
            dataset: default_dataset(),
            gamma: default_gamma(),
            lam: default_lam(),
            max_depth: default_max_depth(),
            rollout_steps: default_rollout_steps(),
            learning_rate: default_learning_rates(),
            quantized: false,
        }
    }
}

impl Config {
    /// Load configuration from the given path.  Supports TOML or JSON based on
    /// the file extension. Returns `None` if parsing fails.
    pub fn from_path(path: &str) -> Option<Self> {
        let Ok(content) = fs::read_to_string(path) else {
            return None;
        };
        if path.ends_with(".json") {
            serde_json::from_str(&content).ok()
        } else {
            toml::from_str(&content).ok()
        }
    }
}
