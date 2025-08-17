mod autograd;
mod data;
mod linear_t;
mod feedforward_t;
mod attention_t;
mod encoder_t;
mod decoder_t;
mod train_backprop;
mod train_elmo;
mod train_noprop;
mod positional;
mod predict;
mod embedding_t;
mod math;         // einfache Matrixops + softmax etc.
mod weights;

use std::env;

fn main() {
    let mut args: Vec<String> = env::args().collect();
    args.push(String::from("backprop"));
    if args.len() < 2 {
        eprintln!("Usage: {} <mode> [optimizer|reload|input]", args[0]);
        eprintln!("Modes: backprop | elmo | noprop | predict");
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
            let input = args.get(2).expect("provide input sentence");
            predict::run(input);
        }
        _ => eprintln!("Unknown mode {}", mode),
    }
}
