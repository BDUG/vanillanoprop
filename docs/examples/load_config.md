# Loading Configuration Example

## Overview

Read training parameters from a `TOML` file and fall back to defaults when the
file is missing.

## Running the Example
=======
This example reads training parameters from a `TOML` file and falls back
to defaults when the file is missing.

**Prerequisites:** `lcm_config.toml` in the repository root.

**Demo command:** (use `cargo run --example`; training binaries run with
`./run.sh`)

```bash
cargo run --example load_config
```

## Explanation

The program prints which values were loaded from `lcm_config.toml`,
illustrating how configuration drives the training setup.

## Next Steps

Learn how these parameters influence optimisation in the
[Training section](../introduction.md#training) of the introduction.
