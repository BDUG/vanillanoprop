# Self-Adapt Agent with Reward Model

## Overview

Demonstrates `SelfAdaptAgent` using a pluggable `RewardModel`.

## Running the Example

A simple 1-gram overlap reward is provided via `NGramReward`.

```bash
./run.sh train-self-adapt --reference "hello" --episodes 1
```

## Explanation

Internally the agent scores each predicted token with the reward model:

```rust
use vanillanoprop::reward::NGramReward;
use vanillanoprop::rl::{LanguageEnv, SelfAdaptAgent};

let env = LanguageEnv::new(b"hello".to_vec());
let reward = NGramReward::new(1);
let mut agent = SelfAdaptAgent::new(env, encoder, decoder, 1e-3, 256, reward);
```

The `NGramReward` returns `1.0` when the predicted token matches the
reference token and `0.0` otherwise. Custom reward hooks can be created
using `ExternalReward` to integrate with arbitrary scoring functions.

## Next Steps

`RewardModel` enables experimenting with different reward functions or
external scorers for self-adaptation.
