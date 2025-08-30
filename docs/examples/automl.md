# AutoML Example

## Overview

The training binaries support a simple random search over hyperparameters.

## Running the Example
=======
**Prerequisites:** a configuration file listing hyperparameters (e.g.,
`config.toml`).

**Training command:** (use `./run.sh`; standalone demos are run with
`cargo run --example`)

```bash
./run.sh train-noprop --auto-ml --config config.toml
```

## Explanation

Candidate learning rates and other options are read from `config.toml`. The
run reports the best score found during the search. To explore additional
hyperparameters, list several values in the configuration file. Each field may
take either a single value or an array of options which are combined during the
search:

```toml
# config.toml
learning_rate = [0.001, 0.01]
batch_size    = [4, 8]
epochs        = [5, 10]
```

`grid_search` tries every combination while `random_search` samples from the
given ranges.

## Next Steps

See the [AutoML section of the introduction](../introduction.md#automl) to
understand how random search fits into the wider training workflow.
