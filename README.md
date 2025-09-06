# vanillanoprop

Vanillanoprop is a vanilla machine learning library implementation started with a wipe coding session.

## Features

- Mixture‑of‑experts layers with a configurable number of experts.
- Training pipelines for NoProp, LCM, Tree Policy Optimization (TreePO) and more.
- Minimal dependencies; builds with stable Rust.
- Configuration via TOML or JSON files with command‑line overrides.
- Collection of examples demonstrating core components.

A more thorough introduction with additional examples can be found in the [documentation index](docs/README.md).

## Supported operating systems

vanillanoprop is tested on Linux, macOS and Windows. Memory tracking utilities
such as `memory::peak_memory_bytes` report actual values on these platforms and
return `0` elsewhere.

## Prerequisites

Examples that train on datasets like MNIST download them on first use and
require an active internet connection. Training binaries read configuration
files such as `noprop_config.toml` or `treepo_config.toml` from the repository
root.

## Installation

Clone the repository and build the project using cargo:

```bash
git clone <repo-url>
cd vanillanoprop
cargo build
```

The `run.sh` script exposes a convenience CLI for training binaries. Standalone
examples under `examples/` are run with `cargo run --example <NAME>`. List the
available training commands with:

```bash
./run.sh
```

### CUDA support

Matrix operations can also run on NVIDIA GPUs. This requires the
[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and a compatible
driver. Build the crate with the optional `cuda` feature:

```bash
cargo build --features cuda
```

When built with this feature a `CudaDevice` type becomes available. Pass it to
`Matrix::matmul_with` to execute the operation on the GPU:

```rust
use vanillanoprop::{device::CudaDevice, math::Matrix};

let a = Matrix::zeros(2, 2);
let b = Matrix::zeros(2, 2);
let dev = CudaDevice::default();
let c = Matrix::matmul_with(&a, &b, &dev);
```

## Usage

Use `./run.sh` for training pipelines and `cargo run --example <NAME>` for
standalone demos. Typical training commands:

```bash
./run.sh train-noprop cnn --moe --num-experts 4
./run.sh predict --moe --num-experts 4
./run.sh train-lcm
./run.sh predict lcm
./run.sh train-treepo
./run.sh train-zero-shot-safe
```

The `--moe` flag enables mixture-of-experts layers and `--num-experts` sets how many experts to use.

When predicting with `--moe`, a corresponding `moe.json` file is read to load the gating and expert weights.

### Web UI

A simple browser interface powered by [Yew](https://yew.rs/) lives under
`web-ui/`. Build the front-end with
[`trunk`](https://trunkrs.dev) and run the web server to serve the compiled
assets:

```bash
cd web-ui
trunk build --release
cd ..
cargo run --bin web_server
```

The helper script `scripts/serve_with_ui.sh` performs these steps and launches
the server.

### Supported models

The library includes multiple built-in models. Training support varies across binaries:

- **Transformer encoder/decoder** – default model for `train-backprop` and `train-noprop`. It is not supported by `train-elmo`, `train-lcm`, `train-resnet`, `train-rnn` or `train-treepo`.
- **Convolutional neural network (CNN)** – pass `cnn` as the model to `train-backprop`, `train-elmo` or `train-noprop`. CNNs are not available in `train-resnet`, `train-rnn`, `train-lcm` or `train-treepo`.
- **Large Concept Model (LCM)** – trained exclusively via `train-lcm`; other training commands do not handle this model.
- **ResNet** – supported only by `train-resnet`. Backprop, Elmo, NoProp, LCM, RNN and TreePO trainings do not support ResNets.
- **Recurrent neural network (RNN)** – trained with `train-rnn` only; it is unsupported in backprop, elmo, noprop, lcm, resnet and treepo binaries.
- **ELMo encoder** – available through `train-elmo`. It cannot be trained with backprop, noprop, lcm, resnet, rnn or treepo commands.
- **Tree Policy Optimization agent** – used by `train-treepo` and does not accept CNN, transformer, LCM, ResNet, RNN or ELMo models.
- **SmolVLM** – `cargo run --example smolvlm` ([walkthrough](docs/examples/smolvlm.md)).
  Downloads a tiny vision‑language model and runs a dummy image + prompt forward pass. Requires internet access.

### Logging

All binaries use the [`log`](https://crates.io/crates/log) facade with
`env_logger` for output. Set the desired verbosity with the new
`--log-level` flag or silence logs with `--quiet`:

```bash
./run.sh train-noprop --log-level debug
./run.sh train-noprop --quiet
```

The `RUST_LOG` environment variable is still honoured when the flag is
omitted.

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

Run a training session with the `train-treepo` command. It loads `treepo_config.toml` by default; pass `--config <FILE>` to use a different configuration:

```bash
./run.sh train-treepo
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

## Zero-shot safe agent

`train-zero-shot-safe` showcases a safety-aware policy that skips updates when the environment reports a failure state. It loads `zero_shot_safe_config.toml` by default; pass `--config <FILE>` to override it:

```bash
./run.sh train-zero-shot-safe
```

The configuration exposes basic hyperparameters:

```
discount_factor = 0.99      # reward discount
safety_thresholds = [-5.0]  # failure when position drops below threshold
learning_rate = 0.1         # policy update rate
rollout_steps = 10          # steps per episode
```

## Configuration

Training parameters such as the number of epochs and batch size can be set via a configuration file in TOML or JSON format. CLI flags override values from the file. Example `config.toml`:

```
epochs = 10
batch_size = 8
```

Example configuration files for each training variant are included in the
repository root and are loaded automatically:

```bash
./run.sh train-backprop
./run.sh train-elmo
./run.sh train-noprop
./run.sh train-lcm
./run.sh train-resnet
./run.sh train-rnn
./run.sh train-treepo
./run.sh train-zero-shot-safe
```

Pass `--config <FILE>` to load a different file or override specific settings from the command line:

```bash
./run.sh train-backprop --config custom.toml
./run.sh train-noprop --epochs 20 --batch-size 16
```

## Examples

Run standalone demos with `cargo run --example <NAME>`; use `./run.sh` for the
training binaries. The `examples` directory contains small programs that
showcase core features of the framework.

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
- **Hugging Face VLM** – `cargo run --example hf_vlm` ([walkthrough](docs/examples/hf_vlm.md))
  Fetches a tiny CLIP-like model and produces embeddings for an image and prompt.

## Hugging Face models

Pretrained Transformers can be loaded directly from the Hugging Face Hub:

```rust
use vanillanoprop::{huggingface, weights, models::TransformerEncoder};

let files = huggingface::fetch_hf_files("bert-base-uncased", None, None)?;
let mut enc = TransformerEncoder::new(/* ... */);
weights::load_transformer_from_hf(&files.config, &files.weights, &mut enc)?;
```

## License

This project is licensed under the [MIT License](LICENSE).

