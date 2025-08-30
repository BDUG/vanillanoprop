# DataLoader Usage

The `DataLoader` trait and struct provide a unified way to work with
datasets that fit in memory.  It supports batching, optional shuffling
and an optional transform hook for preprocessing or data augmentation.

**Prerequisites:** datasets such as MNIST or CIFAR10; they are downloaded on
first use.

This page shows library snippets only. To see a runnable demo, execute
`cargo run --example mnist_cnn`. Training binaries are invoked with `./run.sh`.

```rust
use vanillanoprop::data::{DataLoader, Mnist};

// Create a loader for the MNIST training set with batch size 32
// and shuffling enabled.
let loader = DataLoader::<Mnist>::new(32, true, None);
for batch in loader {
    // `batch` is a `&[(Vec<u8>, usize)]`
    // ... perform training ...
}
```

A transform can be supplied to modify samples before batching:

```rust
use vanillanoprop::data::{DataLoader, Cifar10};

let normalize = |sample: &mut (Vec<u8>, usize)| {
    for p in &mut sample.0 {
        *p = (*p as f32 / 255.0) as u8;
    }
};
let loader = DataLoader::<Cifar10>::new(64, true, Some(Box::new(normalize)));
```
