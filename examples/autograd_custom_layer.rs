use vanillanoprop::config::Config;
use vanillanoprop::tensor::{Node, NodeRef, Tensor};

struct MyLayer {
    weight: NodeRef,
}

impl MyLayer {
    fn new() -> Self {
        // Single weight parameter initialised to 2.0
        let w = Tensor::new(vec![2.0], vec![1, 1]).into_node(true);
        Self { weight: w }
    }

    fn forward(&self, x: &NodeRef) -> NodeRef {
        // Simple linear layer y = x * w
        Node::matmul(x, &self.weight)
    }
}

fn main() {
    // Load configuration and fall back to defaults if the file is missing.
    let _cfg = Config::from_path("configs/autograd_custom_layer.toml").unwrap_or_default();

    let layer = MyLayer::new();
    let x = Tensor::new(vec![3.0], vec![1, 1]).into_node(true);
    let y = layer.forward(&x);
    Node::backward(&y);
    println!("grad x: {:?}", x.borrow().grad.as_ref().unwrap().data);
    println!(
        "grad w: {:?}",
        layer.weight.borrow().grad.as_ref().unwrap().data
    );
}
