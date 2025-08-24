use crate::math::{self, Matrix};

#[derive(Clone)]
pub struct Tensor {
    pub data: Matrix,
}

impl Tensor {
    pub fn from_matrix(data: Matrix) -> Self {
        Tensor { data }
    }

    pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
        let data = a.data.add(&b.data);
        Tensor::from_matrix(data)
    }

    pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
        let data = Matrix::matmul(&a.data, &b.data);
        Tensor::from_matrix(data)
    }

    pub fn transpose(t: &Tensor) -> Tensor {
        Tensor::from_matrix(t.data.transpose())
    }

    pub fn softmax(t: &Tensor) -> Tensor {
        let data = t.data.softmax();
        Tensor::from_matrix(data)
    }

    /// Compute cross-entropy loss and gradient w.r.t. the logits contained in
    /// this tensor.  The returned gradient tensor can be fed directly into a
    /// backward pass of subsequent layers.
    #[allow(dead_code)]
    pub fn softmax_cross_entropy(&self, targets: &[usize], row_offset: usize) -> (f32, Tensor) {
        let (loss, grad, _) = math::softmax_cross_entropy(&self.data, targets, row_offset);
        (loss, Tensor::from_matrix(grad))
    }
}
