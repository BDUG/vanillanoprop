use vanillanoprop::layers::Conv2d;
use vanillanoprop::math::Matrix;

#[test]
fn conv_forward_accepts_valid_square() {
    let mut conv = Conv2d::new(1, 1, 1, 1, 0);
    let x = Matrix::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
    let _ = conv.forward_local(&x);
}

#[test]
#[should_panic(expected = "not divisible by in_channels")]
fn conv_forward_panics_on_channel_mismatch() {
    let mut conv = Conv2d::new(3, 1, 1, 1, 0);
    let x = Matrix::from_vec(1, 7, vec![0.0; 7]);
    let _ = conv.forward_local(&x);
}

#[test]
#[should_panic(expected = "not a perfect square")]
fn conv_forward_panics_on_non_square_input() {
    let mut conv = Conv2d::new(1, 1, 1, 1, 0);
    let x = Matrix::from_vec(1, 3, vec![0.0; 3]);
    let _ = conv.forward_local(&x);
}
