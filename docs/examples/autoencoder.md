# Autoencoder Example

## Overview

Demonstrates a tiny variational autoencoder that reconstructs MNIST images.

## Running the Example

```bash
cargo run --example autoencoder
```

## Explanation

Training prints a reconstruction loss as the model learns to recreate
digits and then writes the weights to disk so the network can be reused.

## Next Steps

Learn more about model persistence in the [Saving section of the
introduction](../introduction.md#saving).
