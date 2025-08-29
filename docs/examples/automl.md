# AutoML Example

The training binaries support a simple random search over hyperparameters.
Enable it via the `--auto-ml` flag:

```bash
./run.sh train-noprop --auto-ml --config config.toml
```

Candidate learning rates and other options are read from `config.toml`.
The run reports the best score found during the search.
