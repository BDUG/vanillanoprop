# vanillanoprop

Vanillanoprop is a vanilla machine learning library implementation.

## Features

- Mixture‑of‑experts layers with a configurable number of experts.
- Training pipelines for NoProp, LCM, Tree Policy Optimization (TreePO) and more.
- Minimal dependencies; builds with stable Rust.
- Configuration via TOML or JSON files with command‑line overrides.
- Collection of examples demonstrating core components.

A more thorough introduction with additional examples can be found in [docs/introduction.md](docs/introduction.md).

## Installation

Clone the repository and build the project using cargo:

```bash
git clone <repo-url>
cd vanillanoprop
cargo build
```

The `run.sh` script exposes a convenience CLI that lists available commands.

```bash
./run.sh
```

## Usage

Typical training commands:

```bash
./run.sh train-noprop cnn --moe --num-experts 4
./run.sh predict --moe --num-experts 4
./run.sh train-lcm
./run.sh predict lcm
./run.sh train-treepo --config treepo_config.toml
```

The `--moe` flag enables mixture-of-experts layers and `--num-experts` sets how many experts to use.

### AutoML

Enable random search over hyperparameters with `--auto-ml` and a config file:

```bash
./run.sh train-noprop --auto-ml --config config.toml
```

See [AutoML](docs/introduction.md#automl) for more details. ([example](docs/examples/automl.md))

### Fine-tuning

Use `--fine-tune <MODEL_ID>` to initialise from a Hugging Face checkpoint and optionally `--freeze-layers <IDX,IDX>` to keep parameters fixed:

```bash
./run.sh train-backprop --fine-tune bert-base-uncased --freeze-layers 0,1,2
```

The example above downloads the `bert-base-uncased` weights, loads them into
the Transformer and updates all parameters except the first three layers.

See [Fine-tuning](docs/introduction.md#fine-tuning) for more details. ([example](docs/examples/fine_tuning.md))
### ONNX export

Training binaries accept an optional `--export-onnx <FILE>` flag. When
provided, the trained weights are exported to an ONNX model after
training completes. Supported layers include linear, convolution,
ReLU, max pooling and batch normalization. The generated model targets
opset 13.

See [ONNX export](docs/introduction.md#onnx-export) for details on the flag. ([example](docs/examples/onnx_export.md))

## Tree Policy Optimization (TreePO)

TreePO combines tree-based planning with policy optimisation to update actions using advantages estimated from a search tree. See the [Tree Policy Optimization paper](https://arxiv.org/abs/2506.03736) for details.

Run a training session with the `train-treepo` command and a required `--config <FILE>` flag:

```bash
./run.sh train-treepo --config treepo_config.toml
```

`treepo_config.toml` introduces several hyperparameters:

```
gamma = 0.99        # discount factor
lam = 0.95          # GAE smoothing
max_depth = 10      # maximum tree expansion depth
rollout_steps = 10  # steps per simulated rollout
episodes = 10       # number of episodes to train
```

During training the binary prints progress for each episode:

```
Episode 1 complete
Episode 2 complete
...
```

See [TreePO](docs/introduction.md#treepo) for more details. ([example](docs/examples/treepo.md))

## Configuration

Training parameters such as the number of epochs and batch size can be set via a configuration file in TOML or JSON format. CLI flags override values from the file. Example `config.toml`:

```
epochs = 10
batch_size = 8
```

Run a training binary with a configuration file:

```bash
./run.sh train-noprop --config config.toml
./run.sh train-lcm --config lcm_config.toml
```

Or override specific settings from the command line:

```bash
./run.sh train-noprop --config config.toml --epochs 20 --batch-size 16
```

## Examples

The `examples` directory contains small programs that showcase core features of the framework.

- **Mixture of Experts** – `cargo run --example mixture_of_experts` ([walkthrough](docs/examples/mixture_of_experts.md))
  Builds a tiny mixture‑of‑experts network and prints gating probabilities.

- **Loading configuration** – `cargo run --example load_config` ([walkthrough](docs/examples/load_config.md))
  Loads training parameters from `lcm_config.toml` and falls back to defaults if the file is not found.

- **MNIST CNN** – `cargo run --example mnist_cnn` ([walkthrough](docs/examples/mnist_cnn.md))
  Trains a simple convolutional network on the MNIST digits.

- **RNN text classification** – `cargo run --example text_rnn` ([walkthrough](docs/examples/text_rnn.md))
  Shows how to build a tiny LSTM classifier on a toy text dataset.

- **Autoencoder** – `cargo run --example autoencoder` ([walkthrough](docs/examples/autoencoder.md))
  Runs a small variational autoencoder to reconstruct MNIST images.
- **Tree policy optimization** – `cargo run --example treepo`
  Demonstrates a minimal `TreePoAgent` on a 1‑D line world. Each episode prints
  the probability of moving toward the goal, which should increase as the agent
  learns.
- **Hugging Face Transformer** – `cargo run --example hf_transformer` ([walkthrough](docs/examples/hf_transformer.md))
  Downloads a tiny BERT model and runs a dummy inference to verify tensor shapes.

## Hugging Face models

Pretrained Transformers can be loaded directly from the Hugging Face Hub:

```rust
use vanillanoprop::{huggingface, weights, models::TransformerEncoder};

let files = huggingface::fetch_hf_files("bert-base-uncased", None)?;
let mut enc = TransformerEncoder::new(/* ... */);
weights::load_transformer_from_hf(&files.config, &files.weights, &mut enc)?;
```

## License

This project is licensed under the [MIT License](LICENSE).

