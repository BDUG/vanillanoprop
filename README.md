# vanillanoprop
Vanilla implementation of no prop.

This crate has been fully adapted to operate on the
[MNIST](http://yann.lecun.com/exdb/mnist/) dataset. The sentence-based
examples have been removed in favour of treating each image as a sequence of
pixel values. A light 3x3 mean convolution is applied to each image during
loading to provide basic convolutional preprocessing. The training modes
(standard backpropagation, NoProp, and an ELMo-inspired method) now all use
these image/label pairs.

The project now also includes a very small convolutional neural network
alongside the original transformer model.  When running predictions you can
choose the model via the command line:

```
./run.sh predict          # uses the CNN (default)
./run.sh predict transformer
```
