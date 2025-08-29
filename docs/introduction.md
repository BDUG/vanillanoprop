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

