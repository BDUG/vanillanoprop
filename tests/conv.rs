use vanillanoprop::layers::{Conv2d, ConvError};
use vanillanoprop::math::Matrix;

#[test]
fn conv_forward_accepts_valid_square() {
    let mut conv = Conv2d::new(1, 1, 1, 1, 0);
    let x = Matrix::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
    assert!(conv.forward_local(&x).is_ok());
}

#[test]
fn conv_forward_errors_on_channel_mismatch() {
    let mut conv = Conv2d::new(3, 1, 1, 1, 0);
    let x = Matrix::from_vec(1, 7, vec![0.0; 7]);
    assert!(matches!(
        conv.forward_local(&x),
        Err(ConvError::ChannelMismatch { .. })
    ));
}

#[test]
fn conv_forward_errors_on_non_square_input() {
    let mut conv = Conv2d::new(1, 1, 1, 1, 0);
    let x = Matrix::from_vec(1, 3, vec![0.0; 3]);
    assert!(matches!(
        conv.forward_local(&x),
        Err(ConvError::NonSquareInput { .. })
    ));
}
