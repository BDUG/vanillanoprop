# TreePO Example

Tree Policy Optimisation (TreePO) combines planning with policy
optimisation.

**Prerequisites:** `treepo_config.toml` in the repository root.

**Training command:** (use `./run.sh`; demos run with `cargo run --example`)

```bash
./run.sh train-treepo --config treepo_config.toml
```

The configuration file sets hyperparameters such as `gamma`, `lam` and
`max_depth`.
