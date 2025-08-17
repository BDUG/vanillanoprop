use crate::math::Matrix;

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
}

