# Hugging Face Transformer Example

## Overview

Fetch pretrained weights from the Hugging Face Hub and run a dummy inference.
Settings, including an optional `hf_token`, are read from
`configs/hf_transformer.toml`.

## Running the Example
=======
Shows how to fetch pretrained weights from the Hugging Face Hub and run a
dummy inference.

**Prerequisites:** internet access to download the model. Include an
`hf_token` in `configs/hf_transformer.toml` if the model requires authentication.
Keep this token out of version control.

**Demo command:** (use `cargo run --example`; training binaries use `./run.sh`)


```bash
cargo run --example hf_transformer
```

## Explanation

The example downloads the model and prints the resulting tensor shapes,
demonstrating how to load external checkpoints and inspect their structure.

## Next Steps

For a full fine-tuning workflow see the
[Fine-tuning section](../introduction.md#fine-tuning) of the introduction.
