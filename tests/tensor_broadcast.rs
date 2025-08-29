use vanillanoprop::tensor::Tensor;

#[test]
fn broadcast_add_works() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let c = Tensor::add(&a, &b);
    assert_eq!(c.shape, vec![2, 3]);
    assert_eq!(c.data, vec![2.0, 4.0, 6.0, 5.0, 7.0, 9.0]);
}

#[test]
fn reshape_and_index() {
    let mut t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_eq!(t.get(&[1, 1]), 4.0);
    t.reshape(vec![4]);
    assert_eq!(t.shape, vec![4]);
    assert_eq!(t.get(&[3]), 4.0);
}

