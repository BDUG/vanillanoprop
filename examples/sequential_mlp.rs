use vanillanoprop::layers::{Activation, FeedForwardT, LinearT, SoftmaxT};
use vanillanoprop::math::Matrix;
use vanillanoprop::models::Sequential;
use vanillanoprop::tensor::Tensor;

fn main() {
    // Build a small multilayer perceptron using the Sequential container
    let mut mlp = Sequential::new();
    mlp.add_layer(Box::new(LinearT::new(4, 16)));
    mlp.add_layer(Box::new(FeedForwardT::new(16, 32, Activation::ReLU)));
    mlp.add_layer(Box::new(LinearT::new(16, 3)));
    mlp.add_layer(Box::new(SoftmaxT::new()));

    // Dummy single-sample input
    let input = Matrix::from_vec(1, 4, vec![0.2, -0.4, 0.1, 0.0]);
    let target_class = 2usize;

    // Forward pass with training to cache values for backward
    let probs = mlp.forward_train(&input);

    // Gradient of cross-entropy loss with respect to the softmax output
    let mut grad = probs.clone();
    for (i, g) in grad.data.iter_mut().enumerate() {
        let class = i % probs.cols;
        *g -= if class == target_class { 1.0 } else { 0.0 };
    }

    // Backward pass and parameter update
    mlp.zero_grad();
    mlp.backward(&grad);
    for layer in mlp.layers.iter_mut() {
        layer.adam_step(0.01, 0.9, 0.999, 1e-8, 0.0);
    }

    // Inference with Tensor input
    let tensor_input = Tensor::from_matrix(input);
    let output = mlp.forward(&tensor_input);
    println!("model output: {:?}", output.data.data);
}
