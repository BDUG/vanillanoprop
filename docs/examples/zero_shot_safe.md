# Zero Shot Safe Example

## Overview

Demonstrates guarding policy updates with a safety check in a toy line world.

## Running the Example
=======
The agent updates its policy only when the environment reports no safety
violations.

**Demo command:** (use `cargo run --example`; training binaries use `./run.sh`)

```bash
cargo run --example zero_shot_safe
```

## Explanation

The environment marks positions less than `-0.5` as failures. When the agent
steps into this region it skips the requested update and falls back to the
recovery policy. Running the program prints the recorded policy and number of
violations:

```
Policy for state 0: Some(1)
Policy for state -1: Some(1)
Safety violations: 1
```

## Next Steps

Experiment with different failure thresholds or recovery strategies to see how
the agent behaves.

