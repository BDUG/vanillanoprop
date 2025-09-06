#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: $0 {download|predict|predict-rnn|train-backprop|train-elmo|train-noprop|train-lcm|train-resnet|train-rnn|train-treepo|train-zero-shot-safe} [model] [opt] [--moe] [--num-experts N] [--resume FILE] [--export-onnx FILE] [--config FILE]

Each training command loads a default configuration file (e.g. configs/backprop_config.toml).
Use --config <FILE> to override it.
USAGE
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

case "$1" in
  download)
    cargo run --bin main -- download
    ;;
  predict)
    shift
    cargo run --bin main -- predict "$@"
    ;;
  predict-rnn)
    shift
    cargo run --bin main -- predict-rnn "$@"
    ;;
  train-backprop)
    shift
    cargo run --bin main -- train-backprop --config configs/backprop_config.toml "$@"
    ;;
  train-elmo)
    shift
    cargo run --bin main -- train-elmo --config configs/elmo_config.toml "$@"
    ;;
  train-noprop)
    shift
    cargo run --bin main -- train-noprop --config configs/noprop_config.toml "$@"
    ;;
  train-lcm)
    shift
    cargo run --bin main -- train-lcm --config configs/lcm_config.toml "$@"
    ;;
  train-resnet)
    shift
    cargo run --bin train_resnet -- --config configs/resnet_config.toml "$@"
    ;;
  train-rnn)
    shift
    cargo run --bin main -- train-rnn --config configs/rnn_config.toml "$@"
    ;;
  train-treepo)
    shift
    cargo run --bin main -- train-treepo --config configs/treepo_config.toml "$@"
    ;;
  train-zero-shot-safe)
    shift
    cargo run --bin main -- train-zero-shot-safe --config configs/zero_shot_safe_config.toml "$@"
    ;;
  *)
    usage
    ;;
esac
