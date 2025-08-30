#!/usr/bin/env bash
set -euo pipefail

# Build the Yew front-end
pushd "$(dirname "$0")/../web-ui" >/dev/null
trunk build --release
popd >/dev/null

# Run the backend server serving the compiled assets
cargo run --bin web_server
