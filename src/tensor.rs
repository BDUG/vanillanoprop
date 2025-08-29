use crate::math;
use crate::math::Matrix;

/// N-dimensional tensor backed by a flat `Vec<f32>`.
///
/// The tensor stores its shape explicitly allowing operations on
/// higher-rank data.  Many legacy parts of the codebase still operate on
/// the 2-D [`Matrix`] type, so conversion helpers are provided to ease the
/// transition.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// Tensor elements in row-major order.
    pub data: Vec<f32>,
    /// Sizes for each dimension.
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor from raw parts.  The number of elements in `data`
    /// must match the product of the requested `shape`.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product());
        Tensor { data, shape }
    }

    /// Convenience constructor taking ownership of a [`Matrix`] while
    /// recording its two dimensional shape.  This is primarily used while
    /// refactoring existing code that still produces matrices.
    pub fn from_matrix(m: Matrix) -> Self {
        Tensor {
            shape: vec![m.rows, m.cols],
            data: m.data,
        }
    }

    /// Compute the flat index for a multi-dimensional coordinate.
    fn offset(&self, idx: &[usize]) -> usize {
        assert_eq!(idx.len(), self.shape.len());
        let mut stride = 1;
        let mut off = 0usize;
        for (i, &dim) in self.shape.iter().rev().enumerate() {
            let id = idx[self.shape.len() - 1 - i];
            assert!(id < dim, "index out of bounds");
            off += id * stride;
            stride *= dim;
        }
        off
    }

    /// Basic immutable indexing.
    pub fn get(&self, idx: &[usize]) -> f32 {
        let off = self.offset(idx);
        self.data[off]
    }

    /// Mutable indexing support.
    pub fn set(&mut self, idx: &[usize], value: f32) {
        let off = self.offset(idx);
        self.data[off] = value;
    }

    /// Change the view of the underlying data without modifying order.
    /// The new shape must contain the same number of elements.
    pub fn reshape(&mut self, new_shape: Vec<usize>) {
        assert_eq!(self.data.len(), new_shape.iter().product());
        self.shape = new_shape;
    }

    /// Broadcast the tensor to a larger shape following numpy semantics
    /// where dimensions of size 1 can be expanded.
    pub fn broadcast_to(&self, target: &[usize]) -> Tensor {
        assert!(target.len() >= self.shape.len());
        for (&src, &dst) in self.shape.iter().rev().zip(target.iter().rev()) {
            assert!(src == dst || src == 1, "cannot broadcast dimension");
        }

        let out_len: usize = target.iter().product();
        let mut out = vec![0.0; out_len];

        // Prepare padded shapes and strides for easier index mapping.
        let mut src_shape = vec![1; target.len()];
        let offset = target.len() - self.shape.len();
        for (i, &dim) in self.shape.iter().enumerate() {
            src_shape[offset + i] = dim;
        }
        let mut src_stride = vec![0; target.len()];
        let mut stride = 1;
        for (i, dim) in src_shape.iter().rev().enumerate() {
            src_stride[src_shape.len() - 1 - i] = stride;
            stride *= *dim;
        }

        for i in 0..out_len {
            // Decode `i` into indices of the target shape and compute the
            // corresponding source index.
            let mut tmp = i;
            let mut src_index = 0usize;
            for ((&t_dim, &s_dim), &s_stride) in target
                .iter()
                .rev()
                .zip(src_shape.iter().rev())
                .zip(src_stride.iter().rev())
            {
                let idx = tmp % t_dim;
                tmp /= t_dim;
                let s_idx = if s_dim == 1 { 0 } else { idx };
                src_index += s_idx * s_stride;
            }
            out[i] = self.data[src_index];
        }

        Tensor {
            data: out,
            shape: target.to_vec(),
        }
    }

    /// Elementwise addition with broadcasting handled by the math helper.
    pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
        math::tensor_add(a, b)
    }

    /// Matrix multiplication treating the last two dimensions as matrices.
    pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
        math::tensor_matmul(a, b)
    }

    /// Transpose a 2-D tensor.
    pub fn transpose(t: &Tensor) -> Tensor {
        math::tensor_transpose(t)
    }

    /// Softmax along the last dimension.
    pub fn softmax(t: &Tensor) -> Tensor {
        math::tensor_softmax(t)
    }

    /// Compute cross-entropy loss and gradient w.r.t. the logits contained in
    /// this tensor.  The returned gradient tensor can be fed directly into a
    /// backward pass of subsequent layers.
    #[allow(dead_code)]
    pub fn softmax_cross_entropy(
        &self,
        targets: &[usize],
        row_offset: usize,
    ) -> (f32, Tensor) {
        let (loss, grad) = math::tensor_softmax_cross_entropy(self, targets, row_offset);
        (loss, grad)
    }
}

