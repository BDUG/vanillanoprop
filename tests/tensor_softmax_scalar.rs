use vanillanoprop::tensor::{Node, Tensor};

#[test]
fn softmax_handles_scalar_input() {
    let x = Tensor::new(vec![0.5], vec![]).into_node(true);
    let s = Node::softmax(&x);
    assert_eq!(s.borrow().value.data, vec![1.0]);
    assert!(s.borrow().value.shape.is_empty());
    Node::backward(&s);
    assert!(x.borrow().grad.is_none());
}
