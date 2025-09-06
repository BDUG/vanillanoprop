use vanillanoprop::config::Config;
use vanillanoprop::math::Matrix;

fn main() {
    let _cfg = Config::from_path("configs/svd.toml").unwrap_or_default();
    let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let (u, s, vt) = m.svd();
    let us = Matrix::matmul(&u, &s);
    let reconstructed = Matrix::matmul(&us, &vt);
    for (a, b) in reconstructed.data.iter().zip(m.data.iter()) {
        assert!((a - b).abs() < 1e-4);
    }
}
