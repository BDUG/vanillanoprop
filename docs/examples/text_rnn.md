# Text RNN Example

## Overview

Build a simple LSTM for toy text classification. The learning rate is loaded
from `configs/text_rnn.toml`.

## Running the Example
=======
This walkthrough builds a simple LSTM for toy text classification.

**Prerequisites:** uses a small in-memory toy dataset.

**Demo command:** (use `cargo run --example`; training binaries use
`./run.sh`)

```bash
cargo run --example text_rnn
```

## Explanation

The program prints the loss for each sample as it trains, illustrating how
inputs are encoded and passed through the network.

## Next Steps

Delve deeper into sequence training in the
[Training section](../introduction.md#training) of the introduction.
