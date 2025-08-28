# Einführung

## Modellaufbau

Mit `MixtureOfExpertsT` lassen sich mehrere Expertennetzwerke kombinieren:

```rust
// examples/mixture_of_experts.rs
let experts: Vec<Box<dyn Layer>> = (0..3)
    .map(|_| Box::new(FeedForwardT::new(4, 8, Activation::ReLU)) as Box<dyn Layer>)
    .collect();
let mut moe = MixtureOfExpertsT::new(4, experts, 1);
```

## Training

Das Training einer RNN für Textklassifikation:

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

## Evaluierung

Einfaches Messen des Verlusts während des MNIST-Trainings:

```rust
// examples/mnist_cnn.rs
let (feat, logits) = cnn.forward(img);
let logits_m = Matrix::from_vec(1, logits.len(), logits);
let (loss, grad, _) = math::softmax_cross_entropy(&logits_m, &[*label as usize], 0);
loss_sum += loss;
// ...
println!("batch {i} loss {}", loss_sum / batch.len() as f32);
```

## Speichern

Nach dem Training kann ein VAE auf die Platte geschrieben werden:

```rust
// src/bin/train_vae.rs
if let Err(e) = save_vae("vae.json", &vae) {
    eprintln!("failed to save model: {e}");
}
```
