# Autoencoder Example

## Overview

Demonstrates a tiny variational autoencoder that reconstructs MNIST
images. Configuration such as batch size and epochs is read from
`configs/autoencoder.toml`.

**Prerequisites:** downloads the MNIST dataset on first run.

**Demo command:** (use `cargo run --example`; training binaries use
`./run.sh`)

```bash
cargo run --example autoencoder
```

## Explanation

Training prints a reconstruction loss as the model learns to recreate
digits and then writes the weights to disk so the network can be reused.

## Next Steps

Learn more about model persistence in the [Saving section of the
introduction](../introduction.md#saving).
