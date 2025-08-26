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
        "predict" => {
            let mut model: Option<String> = None;
            let mut moe = false;
            let mut num_experts = 1usize;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--moe" => moe = true,
                    "--num-experts" => {
                        if i + 1 < args.len() {
                            num_experts = args[i + 1].parse().unwrap_or(1);
                            i += 1;
                        }
                    }
                    other => {
                        if !other.starts_with("--") && model.is_none() {
                            model = Some(other.to_string());
                        }
                    }
                }
                i += 1;
            }
            predict::run(model.as_deref(), moe, num_experts);
        }
        "download" => data::download_mnist(),
        other => eprintln!("Unknown mode {}", other),
    }
}
