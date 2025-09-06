# Hugging Face VLM Example

## Overview

Download a tiny vision-language model from the Hugging Face Hub and run a dummy image + text forward pass. The example reads
settings such as the optional Hugging Face token from `configs/hf_vlm.toml`.

## Running the Example
Fetches a small CLIP-like checkpoint and produces embeddings for an image and a prompt.

**Prerequisites:** internet access to download the model. Add an `hf_token`
to `configs/hf_vlm.toml` when authentication is needed and avoid committing
files containing the token.

**Demo command:** (use `cargo run --example`; training binaries use `./run.sh`)

```bash
cargo run --example hf_vlm
```

## Explanation

The program loads both the vision and text components, illustrating how to combine image and text encoders.

## Next Steps

See the [Hugging Face Transformer example](hf_transformer.md) for a walkthrough focused on text-only models.

