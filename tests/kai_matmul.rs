#![cfg(all(target_arch = "aarch64", feature = "kai"))]

use vanillanoprop::device::{Cpu, KaiDevice};
use vanillanoprop::math::Matrix;

fn compare(a: Matrix, b: Matrix) {
    let cpu = Cpu;
    let kai = KaiDevice;
    let cpu_res = Matrix::matmul_with(&a, &b, &cpu);
    let kai_res = Matrix::matmul_with(&a, &b, &kai);
    for (x, y) in cpu_res.data.iter().zip(kai_res.data.iter()) {
        assert!((x - y).abs() < 1e-5);
    }
}

#[test]
fn matmul_square() {
    let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
    compare(a, b);
}

#[test]
fn matmul_rectangular() {
    let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    compare(a, b);
}
