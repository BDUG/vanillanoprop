use crate::tensor::Tensor;

/// A simple fusion layer that combines encoder outputs by concatenation.
///
/// All inputs must have the same shape except for the last dimension which
/// is concatenated together in the output tensor.
pub struct Fusion;

impl Fusion {
    /// Concatenate the encoder outputs along the final dimension.
    pub fn concat(cnn: &Tensor, rnn: &Tensor, transformer: &Tensor) -> Tensor {
        assert_eq!(cnn.shape.len(), rnn.shape.len());
        assert_eq!(cnn.shape.len(), transformer.shape.len());
        for i in 0..cnn.shape.len() - 1 {
            assert_eq!(cnn.shape[i], rnn.shape[i]);
            assert_eq!(cnn.shape[i], transformer.shape[i]);
        }
        let last = cnn.shape[cnn.shape.len() - 1]
            + rnn.shape[rnn.shape.len() - 1]
            + transformer.shape[transformer.shape.len() - 1];
        let mut shape = cnn.shape.clone();
        *shape.last_mut().unwrap() = last;
        let mut data = Vec::with_capacity(cnn.data.len() + rnn.data.len() + transformer.data.len());
        data.extend_from_slice(&cnn.data);
        data.extend_from_slice(&rnn.data);
        data.extend_from_slice(&transformer.data);
        Tensor::new(data, shape)
    }
}
