#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {download|predict|train-backprop|train-elmo|train-noprop|train-lcm|train-resnet} [model] [opt] [--moe] [--num-experts N]" >&2
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
  *)
    usage
    ;;
esac
