#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {download|predict|predict-rnn|train-backprop|train-elmo|train-noprop|train-lcm|train-resnet|train-rnn|train-treepo} [model] [opt] [--moe] [--num-experts N] [--resume FILE]" >&2
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
    cargo run --bin main -- train-backprop "$@"
    ;;
  train-elmo)
    shift
    cargo run --bin main -- train-elmo "$@"
    ;;
  train-noprop)
    shift
    cargo run --bin main -- train-noprop "$@"
    ;;
  train-lcm)
    shift
    cargo run --bin main -- train-lcm "$@"
    ;;
  train-resnet)
    shift
    cargo run --bin train_resnet -- "$@"
    ;;
  train-rnn)
    shift
    cargo run --bin main -- train-rnn "$@"
    ;;
  train-treepo)
    shift
    cargo run --bin main -- train-treepo "$@"
    ;;
  *)
    usage
    ;;
esac
