# Mixture of Experts Example

This walkthrough builds a tiny mixture-of-experts layer and prints the
routing probabilities for each input. Run it with cargo:

```bash
cargo run --example mixture_of_experts
```

The source code demonstrates how to construct experts and combine them
with the `MixtureOfExpertsT` layer.
