use vanillanoprop::math::Matrix;

#[test]
fn svd_reconstructs_matrix() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
    let (u, s, vt) = m.svd();
    let us = Matrix::matmul(&u, &s);
    let recon = Matrix::matmul(&us, &vt);
    for (a, b) in recon.data.iter().zip(m.data.iter()) {
        assert!((a - b).abs() < 1e-4, "{} vs {}", a, b);
    }
}
