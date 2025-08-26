use crate::math::Matrix;
use crate::tensor::Tensor;

/// Apply ReLU activation in place on a tensor.
pub fn forward_tensor(t: &mut Tensor) {
    for v in t.data.data.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Apply ReLU activation in place on a matrix and return a mask for backward.
pub fn forward_matrix(m: &mut Matrix) -> Vec<f32> {
    let mut mask = vec![0.0; m.data.len()];
    for (i, v) in m.data.iter_mut().enumerate() {
        if *v < 0.0 {
            *v = 0.0;
        } else {
            mask[i] = 1.0;
        }
    }
    mask
}

/// Apply the stored ReLU mask to the gradient matrix.
pub fn backward(grad: &mut Matrix, mask: &[f32]) {
    for (g, &m) in grad.data.iter_mut().zip(mask.iter()) {
        *g *= m;
    }
}

