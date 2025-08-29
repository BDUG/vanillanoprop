# AutoML Example

The training binaries support a simple random search over hyperparameters.
Enable it via the `--auto-ml` flag:

```bash
./run.sh train-noprop --auto-ml --config config.toml
```

Candidate learning rates and other options are read from `config.toml`.
The run reports the best score found during the search.

To explore additional hyperparameters, list several values in the
configuration file. Each field may take either a single value or an array of
options which are combined during the search:

```toml
# config.toml
learning_rate = [0.001, 0.01]
batch_size    = [4, 8]
epochs        = [5, 10]
```

`grid_search` will try every combination while `random_search` samples from the
given ranges.
