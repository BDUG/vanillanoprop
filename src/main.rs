mod autograd;
mod data;
mod decoding;
mod math; // einfache Matrixops + softmax etc.
mod metrics;
mod positional;
mod predict;
mod train_backprop;
mod train_elmo;
mod train_noprop;
mod transformer_t;
mod weights;

use std::env;

fn main() {
    let mut args: Vec<String> = env::args().collect();
    args.push(String::from("backprop"));
    if args.len() < 2 {
        eprintln!("Usage: {} <mode> [optimizer|reload]", args[0]);
        eprintln!("Modes: backprop | elmo | noprop | predict | download");
        return;
    }

    let mode = args[1].as_str();
    match mode {
        "backprop" => {
            let opt = args.get(2).map(|s| s.as_str()).unwrap_or("sgd");
            train_backprop::run(opt);
        }
        "elmo" => {
            let reload_flag = args.get(2).map(|s| s == "reload").unwrap_or(false);
            if reload_flag {
                // fine-tune logic would go here
                println!("(Reload for train_elmo not implemented in this demo)");
            }
            train_elmo::run();
        }
        "noprop" => train_noprop::run(),
        "predict" => {
            predict::run();
        }
        "download" => {
            data::download_mnist();
        }
        _ => eprintln!("Unknown mode {}", mode),
    }
}
