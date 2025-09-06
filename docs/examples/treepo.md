# TreePO Example

## Overview

Tree Policy Optimisation (TreePO) combines planning with policy optimisation.

## Running the Example
=======
Tree Policy Optimisation (TreePO) combines planning with policy
optimisation.

**Prerequisites:** `configs/treepo.toml` in the repository.

**Training command:** (use `./run.sh`; demos run with `cargo run --example`)


```bash
cargo run --example treepo
```

## Explanation

The configuration file sets hyperparameters such as `gamma`, `lam` and
`max_depth`. Running the command produces reward and policy update logs while
the agent learns.

## Next Steps

For more on the algorithm and configuration see the
[TreePO section](../introduction.md#treepo) of the introduction.
