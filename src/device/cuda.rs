use crate::math::{matmul_cuda, Matrix};
use super::Device;

/// [`Device`] implementation that dispatches matrix operations to CUDA kernels
/// via the [`cust`](https://crates.io/crates/cust) crate.
///
/// This type is only available when the crate is built with the `cuda` feature.
#[derive(Default, Clone, Copy)]
pub struct CudaDevice;

impl Device for CudaDevice {
    fn matmul(&self, a: &Matrix, b: &Matrix) -> Matrix {
        matmul_cuda(a, b)
    }
}
