use std::process::Command;
use vanillanoprop::{data, predict, DatasetKind};

mod common;

fn main() {
    let args = common::init_logging();
    if args.len() < 2 {
        eprintln!("Usage: {} [--log-level <LEVEL>|--quiet] <mode>", args[0]);
        eprintln!(
            "Modes: predict | predict-rnn | download | train-backprop | train-elmo | train-noprop | train-lcm | train-rnn | train-treepo | train-zero-shot-safe | automl",
        );
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
                _export_onnx,
                _fine_tune,
                _freeze_layers,
                _auto_ml,
                _config,
                positional,
            ) = common::parse_cli(args[2..].iter().cloned());
            let model_opt = if positional.is_empty() {
                None
            } else {
                Some(model.as_str())
            };
            let res = predict::run(DatasetKind::Mnist, model_opt, moe, num_experts);
            println!("{}", res);
        }
        "predict-rnn" => {
            let (
                _model,
                _opt,
                moe,
                num_experts,
                _lr_cfg,
                _resume,
                _save_every,
                _ckpt_dir,
                _log_dir,
                _experiment,
                _export_onnx,
                _fine_tune,
                _freeze_layers,
                _auto_ml,
                _config,
                _positional,
            ) = common::parse_cli(args[2..].iter().cloned());
            let res = predict::run(DatasetKind::Mnist, Some("rnn"), moe, num_experts);
            println!("{}", res);
        }
        "download" => data::download_mnist(),
        "train-backprop"
        | "train-elmo"
        | "train-noprop"
        | "train-lcm"
        | "train-rnn"
        | "train-treepo"
        | "train-zero-shot-safe" => {
            let bin = match args[1].as_str() {
                "train-backprop" => "train_backprop",
                "train-elmo" => "train_elmo",
                "train-lcm" => "train_lcm",
                "train-rnn" => "train_rnn",
                "train-treepo" => "train_treepo",
                "train-zero-shot-safe" => "train_zero_shot_safe",
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
        "automl" => {
            let status = Command::new("cargo")
                .args(["run", "--bin", "train_noprop", "--"])
                .args(&args[2..])
                .arg("--auto-ml")
                .status()
                .expect("failed to run automl");
            if !status.success() {
                eprintln!("automl exited with status: {:?}", status.code());
            }
        }
        other => eprintln!("Unknown mode {}", other),
    }
}
