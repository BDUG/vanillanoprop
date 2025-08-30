# Mixture of Experts Example

This walkthrough builds a tiny mixture-of-experts layer and prints the
routing probabilities for each input.

**Prerequisites:** none beyond the standard Rust toolchain.

**Demo command:** (use `cargo run --example`; training binaries use
`./run.sh`)

```bash
cargo run --example mixture_of_experts
```

The source code demonstrates how to construct experts and combine them
with the `MixtureOfExpertsT` layer.
