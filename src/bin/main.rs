use std::env;
use vanillanoprop::{data, predict};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <mode>", args[0]);
        eprintln!("Modes: predict | download");
        return;
    }

    match args[1].as_str() {
        "predict" => predict::run(),
        "download" => data::download_mnist(),
        other => eprintln!("Unknown mode {}", other),
    }
}
