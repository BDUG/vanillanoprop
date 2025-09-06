use vanillanoprop::config::Config;

fn main() {
    // Load configuration from a file. Falls back to defaults if unavailable.
    let cfg = Config::from_path("configs/load_config.toml").unwrap_or_default();
    println!("epochs: {} batch_size: {}", cfg.epochs, cfg.batch_size);
}
