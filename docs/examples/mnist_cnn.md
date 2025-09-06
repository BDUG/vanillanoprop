# MNIST CNN Example

## Overview

Train a small convolutional network on the MNIST digits dataset. Hyperparameters
such as `batch_size`, `epochs`, and `learning_rate` are loaded from
`configs/mnist_cnn.toml`.

## Running the Example
=======
Trains a small convolutional network on the MNIST digits dataset.

**Prerequisites:** downloads the MNIST dataset on first run.

**Demo command:** (use `cargo run --example`; training binaries use `./run.sh`)

```bash
cargo run --example mnist_cnn
```

## Explanation

The program prints the loss after each batch, giving a quick signal of training
progress and helping you monitor convergence.

## Next Steps

Learn about interpreting these metrics in the
[Evaluation section](../introduction.md#evaluation) of the introduction.
