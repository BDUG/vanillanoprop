# Autoencoder Example

Demonstrates a tiny variational autoencoder that reconstructs MNIST
images.

**Prerequisites:** downloads the MNIST dataset on first run.

**Demo command:** (use `cargo run --example`; training binaries use
`./run.sh`)

```bash
cargo run --example autoencoder
```

Weights are saved to disk after training so the model can be reused
later.
