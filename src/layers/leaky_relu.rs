use crate::math::Matrix;
use crate::tensor::Tensor;

const SLOPE: f32 = 0.01;

/// Apply leaky ReLU activation in place on a tensor.
pub fn forward_tensor(t: &mut Tensor) {
    for v in t.data.iter_mut() {
        if *v < 0.0 {
            *v *= SLOPE;
        }
    }
}

/// Apply leaky ReLU activation in place on a matrix and return derivative mask.
pub fn forward_matrix(m: &mut Matrix) -> Vec<f32> {
    let mut mask = vec![0.0; m.data.len()];
    for (i, v) in m.data.iter_mut().enumerate() {
        if *v < 0.0 {
            *v *= SLOPE;
            mask[i] = SLOPE;
        } else {
            mask[i] = 1.0;
        }
    }
    mask
}

/// Apply derivative mask to gradient matrix.
pub fn backward(grad: &mut Matrix, mask: &[f32]) {
    for (g, &m) in grad.data.iter_mut().zip(mask.iter()) {
        *g *= m;
    }
}
