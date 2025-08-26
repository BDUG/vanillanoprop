use crate::math::{matmul_cpu, Matrix};

/// Abstraction over a compute device capable of executing matrix operations.
pub trait Device {
    /// Multiply two matrices returning the result on the same device.
    fn matmul(&self, a: &Matrix, b: &Matrix) -> Matrix;
}

/// Default CPU implementation of [`Device`].
#[derive(Default, Clone, Copy)]
pub struct Cpu;

impl Device for Cpu {
    fn matmul(&self, a: &Matrix, b: &Matrix) -> Matrix {
        matmul_cpu(a, b)
    }
}
