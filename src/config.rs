use serde::Deserialize;
use std::fs;

/// Training configuration loaded from a TOML or JSON file.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Number of training epochs.
    pub epochs: usize,
    /// Batch size used when loading data.
    pub batch_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self { epochs: 5, batch_size: 4 }
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
