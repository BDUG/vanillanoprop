# vanillanoprop
Vanilla implementation of no prop.

This crate has been fully adapted to operate on the
[MNIST](http://yann.lecun.com/exdb/mnist/) dataset. The sentence-based
examples have been removed in favour of treating each image as a sequence of
pixel values. The training modes (standard backpropagation, NoProp, and an
ELMo-inspired method) now all use these image/label pairs.
