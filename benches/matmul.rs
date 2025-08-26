use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use vanillanoprop::math::Matrix;

fn matmul_naive(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.rows);
    let mut out = vec![0.0; a.rows * b.cols];
    for i in 0..a.rows {
        let a_row = &a.data[i * a.cols..(i + 1) * a.cols];
        for k in 0..a.cols {
            let a_val = a_row[k];
            let b_row = &b.data[k * b.cols..(k + 1) * b.cols];
            for j in 0..b.cols {
                out[i * b.cols + j] += a_val * b_row[j];
            }
        }
    }
    Matrix::from_vec(a.rows, b.cols, out)
}

fn bench_matmul(c: &mut Criterion) {
    let size = 256;
    let mut rng = rand::thread_rng();
    let a_data: Vec<f32> = (0..size * size).map(|_| rng.gen()).collect();
    let b_data: Vec<f32> = (0..size * size).map(|_| rng.gen()).collect();
    let a = Matrix::from_vec(size, size, a_data);
    let b = Matrix::from_vec(size, size, b_data);

    c.bench_function("matmul_naive", |bencher| {
        bencher.iter(|| {
            let res = matmul_naive(black_box(&a), black_box(&b));
            black_box(res);
        });
    });

    c.bench_function("matmul_blocked", |bencher| {
        bencher.iter(|| {
            let res = Matrix::matmul(black_box(&a), black_box(&b));
            black_box(res);
        });
    });
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
