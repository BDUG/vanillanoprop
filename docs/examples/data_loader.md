# DataLoader Usage

## Overview

The `DataLoader` trait and struct provide a unified way to work with datasets
that fit in memory. It supports batching, optional shuffling and an optional
transform hook for preprocessing or data augmentation.

## Running the Example

Embed the following snippet in a small binary crate and execute it with
`cargo run`:

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

## Explanation

Iterating over the loader yields mini-batches for training. A transform can be
supplied to modify samples before batching:

```rust
use vanillanoprop::data::{DataLoader, Cifar10};

let normalize = |sample: &mut (Vec<u8>, usize)| {
    for p in &mut sample.0 {
        *p = (*p as f32 / 255.0) as u8;
    }
};
let loader = DataLoader::<Cifar10>::new(64, true, Some(Box::new(normalize)));
```

## Next Steps

See how batches feed into training loops in the
[Training section](../introduction.md#training) of the introduction.
