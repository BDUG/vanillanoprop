# Introduction

This guide introduces the core building blocks of the `vanillanoprop` framework and shows how to compose models, train networks, evaluate performance and persist learned parameters. Each section references an executable example to explore the concepts further.

## Model Composition

`MixtureOfExpertsT` combines several expert networks into a single layer:

```rust
// examples/mixture_of_experts.rs
let experts: Vec<Box<dyn Layer>> = (0..3)
    .map(|_| Box::new(FeedForwardT::new(4, 8, Activation::ReLU)) as Box<dyn Layer>)
    .collect();
let mut moe = MixtureOfExpertsT::new(4, experts, 1);
```

The gating mechanism selects which expert should process each input, enabling sparse routing and efficient scaling.

Weights for a mixture-of-experts layer can be persisted and later reloaded:

```rust
use vanillanoprop::weights::{save_moe, load_moe};
save_moe("moe.json", &mut moe)?;
let restored = load_moe("moe.json", 4, 8, 3)?;
```

## Training

The following snippet trains a recurrent neural network for text classification:

```rust
// examples/text_rnn.rs
for (seq, label) in batch {
    let mut mat = Matrix::zeros(seq.len(), 1);
    for (t, token) in seq.iter().enumerate() {
        mat.set(t, 0, token[0]);
    }
    let logits = rnn.forward_train(&mat);
    let (loss, grad, _) = math::softmax_cross_entropy(&logits, &[*label as usize], 0);
    rnn.zero_grad();
    rnn.backward(&grad);
    rnn.adam_step(0.05, 0.9, 0.999, 1e-8, 0.0);
}
```

It constructs an input matrix, performs a forward pass, computes the loss and gradients, and updates the weights with Adam.

## Evaluation

During MNIST training you can monitor the loss like this:

```rust
// examples/mnist_cnn.rs
let (feat, logits) = cnn.forward(img);
let logits_m = Matrix::from_vec(1, logits.len(), logits);
let (loss, grad, _) = math::softmax_cross_entropy(&logits_m, &[*label as usize], 0);
loss_sum += loss;
// ...
println!("batch {i} loss {}", loss_sum / batch.len() as f32);
```

Aggregating the loss across batches provides a simple sanity check for overfitting or convergence issues.

## Saving

After training, a variational autoencoder can be written to disk:

```rust
// src/bin/train_vae.rs
if let Err(e) = save_vae("vae.json", &vae) {
    eprintln!("failed to save model: {e}");
}
```

Persisting models allows you to reload them later for inference or continued training.


## AutoML

```rust
// src/bin/train_noprop.rs
if auto_ml {
    use vanillanoprop::automl::{random_search, SearchSpace};
    let space = SearchSpace::from_config(&config);
    let eval = |_cfg: Config| rand::random::<f32>();
    let (_best_cfg, best_score) = random_search(&space, 10, eval, &mut rng, &mut logger);
    println!("AutoML best score: {best_score:.4}");
    return;
}
```

Enable random search with `./run.sh train-noprop --auto-ml --config config.toml`.
See the [AutoML example](examples/automl.md) for a full walkthrough.  Multiple
values for fields such as `learning_rate`, `batch_size` or `epochs` may be
listed in the configuration to let the search explore different
combinations.

## Fine-tuning

```rust
// src/bin/train_noprop.rs
let _ft = fine_tune.map(|model_id| {
    vanillanoprop::fine_tune::run(&model_id, freeze_layers, |_, _| Ok(()))
        .expect("fine-tune load failed")
});
```

Initialise from a Hugging Face checkpoint using
`./run.sh train-backprop --fine-tune bert-base-uncased`.
Further details in the [fine-tuning example](examples/fine_tuning.md).

## ONNX export

```rust
// src/bin/common.rs
"--export-onnx" => {
    if let Some(v) = args.next() {
        export_onnx = Some(v);
    }
}
```

Training binaries accept `--export-onnx <FILE>` to write weights in the
ONNX format. The exporter currently handles linear, convolution, ReLU,
max pooling and batch normalization layers. Check the
[ONNX export example](examples/onnx_export.md).

## TreePO

```rust
// src/rl/treepo.rs
pub fn update_policy(&mut self) {
    // Backpropagate advantages and update policy with learning rate `lr`
}
```

`TreePoAgent::new` expects an additional `lr` parameter controlling the step
size of the policy update.

Launch a TreePO run with `./run.sh train-treepo --config treepo_config.toml`.
See the [TreePO example](examples/treepo.md) for configuration details.
