# Hugging Face Transformer Example

Shows how to fetch pretrained weights from the Hugging Face Hub and run a
dummy inference.

**Prerequisites:** internet access to download the model.

**Demo command:** (use `cargo run --example`; training binaries use `./run.sh`)

```bash
cargo run --example hf_transformer
```

The example downloads the model and prints the resulting tensor shapes.
