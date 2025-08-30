# Loading Configuration Example

This example reads training parameters from a `TOML` file and falls back
to defaults when the file is missing.

**Prerequisites:** `lcm_config.toml` in the repository root.

**Demo command:** (use `cargo run --example`; training binaries run with
`./run.sh`)

```bash
cargo run --example load_config
```

Review the output to see which values were loaded from `lcm_config.toml`.
