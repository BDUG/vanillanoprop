#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {download|predict|train-backprop|train-elmo|train-noprop} [model]" >&2
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
      cargo run --bin train_backprop "$@"
      ;;
    train-elmo)
      shift
      cargo run --bin train_elmo "$@"
      ;;
    train-noprop)
      shift
      cargo run --bin train_noprop "$@"
      ;;
  *)
    usage
    ;;
esac
