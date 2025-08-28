# vanillanoprop

This here is a vanilla implementation of [NoProp](https://arxiv.org/html/2503.24322v2) according to *Qinyu Li*.
It is part of a baremetal AI framework, not requering exhausting 3rd party libraries, compare *Cargo.toml*. 

Call
```
./run.sh
```
to see the continously evolving command line parameters.

For example:
```
./run.sh train-noprop cnn --moe --num-experts 4
./run.sh predict --moe --num-experts 4
./run.sh train-lcm
./run.sh predict lcm
./run.sh train-treepo --config treepo_config.toml
```
The `--moe` flag enables mixture-of-experts layers and `--num-experts` sets
how many experts to use.

## Tree Policy Optimization (TreePO)

TreePO combines tree-based planning with policy optimisation to update
actions using advantages estimated from a search tree. A full description of
the method can be found in the [Tree Policy Optimization paper](https://arxiv.org/abs/2506.03736).

Run a training session with a configuration file:

```
./run.sh train-treepo --config treepo_config.toml
```

### TreePO configuration

`treepo_config.toml` introduces several hyperparameters:

```
gamma = 0.99        # discount factor
lam = 0.95          # GAE smoothing
max_depth = 10      # maximum tree expansion depth
rollout_steps = 10  # steps per simulated rollout
episodes = 10       # number of episodes to train
```

During training the binary prints progress for each episode, e.g.:

```
Episode 1 complete
Episode 2 complete
...
```

## Configuration

Training parameters such as the number of epochs and batch size can be set via
a configuration file in TOML or JSON format. CLI flags override values from the
file. Example `config.toml`:

```
epochs = 10
batch_size = 8
```

Run a training binary with a configuration file:

```
./run.sh train-noprop --config config.toml
./run.sh train-lcm --config lcm_config.toml
```

Or override specific settings from the command line:

```
./run.sh train-noprop --config config.toml --epochs 20 --batch-size 16
```

## Examples

The `examples` directory contains small programs that showcase core features of the framework.

### Mixture of Experts

```bash
cargo run --example mixture_of_experts
```

Builds a tiny mixture-of-experts network and prints the gating probabilities to demonstrate sparse expert routing.

### Loading configuration

```bash
cargo run --example load_config
```

Loads training parameters from `lcm_config.toml` and falls back to defaults if the file is not found.

### MNIST CNN

```bash
cargo run --example mnist_cnn
```

Trains a simple convolutional network on the MNIST digits using the dataset utilities.

### RNN text classification

```bash
cargo run --example text_rnn
```

Shows how to build a tiny LSTM classifier on a toy text dataset.

### Autoencoder

```bash
cargo run --example autoencoder
```

Runs a small variational autoencoder to reconstruct MNIST images.
