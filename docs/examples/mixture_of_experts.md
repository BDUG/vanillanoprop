# Mixture of Experts Example

## Overview

Build a tiny mixture-of-experts layer and inspect the routing probabilities for
each input.

## Running the Example
=======
This walkthrough builds a tiny mixture-of-experts layer and prints the
routing probabilities for each input.

**Prerequisites:** none beyond the standard Rust toolchain.

**Demo command:** (use `cargo run --example`; training binaries use
`./run.sh`)

```bash
cargo run --example mixture_of_experts
```

## Explanation

The source code demonstrates how to construct experts and combine them with the
`MixtureOfExpertsT` layer, printing the gating probabilities for analysis.

## Next Steps

Read about composing complex models in the
[Model Composition section](../introduction.md#model-composition) of the
introduction.
