use crate::math::Matrix;
use crate::tensor::Tensor;

/// Apply tanh activation in place on a tensor.
pub fn forward_tensor(t: &mut Tensor) {
    for v in t.data.data.iter_mut() {
        *v = v.tanh();
    }
}

/// Apply tanh activation in place on a matrix.
pub fn forward_matrix(m: &mut Matrix) {
    for v in m.data.iter_mut() {
        *v = v.tanh();
    }
}

/// Multiply gradient with derivative of tanh using activated values.
pub fn backward(grad: &mut Matrix, activated: &Matrix) {
    for (g, &h) in grad.data.iter_mut().zip(activated.data.iter()) {
        *g *= 1.0 - h * h;
    }
}
