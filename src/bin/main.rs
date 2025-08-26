use std::{env, process::Command};
use vanillanoprop::{data, predict};

mod common;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <mode>", args[0]);
        eprintln!("Modes: predict | download | train-backprop | train-elmo | train-noprop");
        return;
    }

    match args[1].as_str() {
        "predict" => {
            let (
                model,
                _opt,
                moe,
                num_experts,
                _lr_cfg,
                _resume,
                _save_every,
                _ckpt_dir,
                _log_dir,
                _experiment,
                _config,
                positional,
            ) = common::parse_cli(args[2..].iter().cloned());
            let model_opt = if positional.is_empty() {
                None
            } else {
                Some(model.as_str())
            };
            predict::run(model_opt, moe, num_experts);
        }
        "download" => data::download_mnist(),
        "train-backprop" | "train-elmo" | "train-noprop" => {
            let bin = match args[1].as_str() {
                "train-backprop" => "train_backprop",
                "train-elmo" => "train_elmo",
                _ => "train_noprop",
            };
            let status = Command::new("cargo")
                .args(["run", "--bin", bin, "--"])
                .args(&args[2..])
                .status()
                .expect("failed to run training binary");
            if !status.success() {
                eprintln!("{} exited with status: {:?}", bin, status.code());
            }
        }
        other => eprintln!("Unknown mode {}", other),
    }
}
