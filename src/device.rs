use crate::math::{matmul_cpu, Matrix};

#[cfg(target_arch = "aarch64")]
use crate::math::{inc_add_ops_by, inc_mul_ops_by};
#[cfg(target_arch = "aarch64")]
use crate::ffi::kai::kai_matmul;

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

/// [`Device`] backed by the Kai microkernel over FFI.
#[cfg(target_arch = "aarch64")]
#[derive(Default, Clone, Copy)]
pub struct KaiDevice;

#[cfg(target_arch = "aarch64")]
impl Device for KaiDevice {
    fn matmul(&self, a: &Matrix, b: &Matrix) -> Matrix {
        assert_eq!(a.cols, b.rows);
        let m = a.rows;
        let n = b.cols;
        let k_dim = a.cols;

        let muls = m * k_dim * n;
        let adds = muls;
        inc_mul_ops_by(muls);
        inc_add_ops_by(adds);

        let mut out = Matrix::zeros(m, n);
        unsafe {
            kai_matmul(
                a.data.as_ptr(),
                b.data.as_ptr(),
                out.data.as_mut_ptr(),
                m as libc::size_t,
                n as libc::size_t,
                k_dim as libc::size_t,
            );
        }
        out
    }
}
