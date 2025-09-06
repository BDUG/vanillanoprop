# SmolVLM Example

## Overview

Run a minimal vision-language model that mirrors SmolVLM-Instruct using the new loader. Configuration, including an optional
`hf_token`, is read from `configs/smolvlm.toml`.

## Running the Example

Downloads a small checkpoint and performs a dummy image + prompt forward pass.

**Prerequisites:** internet access to fetch model weights.

**Demo command:** (use `cargo run --example`; training binaries use `./run.sh`)

```bash
cargo run --example smolvlm
```

## Explanation

The example constructs a [`SmolVLM`](../../src/models/smolvlm.rs) instance, loads weights from the Hugging Face Hub and fuses image and text embeddings.

## Next Steps

See the [Hugging Face VLM example](hf_vlm.md) for a CLIP-style model.

