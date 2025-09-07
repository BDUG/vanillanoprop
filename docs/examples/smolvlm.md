# SmolVLM Example

## Overview

Run a minimal vision-language model that mirrors SmolVLM-Instruct using the new loader. Configuration, including an optional
`hf_token`, is read from `configs/smolvlm.toml` and a `tokenizer.json` is fetched and parsed by the internal tokenizer so the model's output can be decoded.

## Running the Example

Downloads a small checkpoint and performs an image + prompt forward pass. The `tokenizer.json` vocabulary is loaded by the built-in tokenizer and used to map model outputs back to readable text.

**Prerequisites:**

- Internet access to fetch model weights.
- Compile with `--features vlm` to pull in the `image` dependency.

**Demo command:**

```bash
cargo run --example smolvlm --features vlm path/to/image.png
```

## Explanation

The example constructs a [`SmolVLM`](../../src/models/smolvlm.rs) instance, loads weights and a tokenizer JSON file from the Hugging Face Hub, fuses image and text embeddings, then applies an `argmax` over the vocabulary dimension to obtain token ids which are decoded to text by the internal tokenizer.

## Next Steps

See the [Hugging Face VLM example](hf_vlm.md) for a CLIP-style model.

