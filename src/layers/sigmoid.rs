use crate::math::Matrix;
use crate::tensor::Tensor;

/// Apply sigmoid activation in place on a tensor.
pub fn forward_tensor(t: &mut Tensor) {
    for v in t.data.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

/// Apply sigmoid activation in place on a matrix.
pub fn forward_matrix(m: &mut Matrix) {
    for v in m.data.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

/// Multiply gradient with derivative of sigmoid using activated values.
pub fn backward(grad: &mut Matrix, activated: &Matrix) {
    for (g, &h) in grad.data.iter_mut().zip(activated.data.iter()) {
        *g *= h * (1.0 - h);
    }
}

