use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use vanillanoprop::math::tensor_matmul;
use vanillanoprop::tensor::Tensor;

fn tensor_matmul_naive(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape.len(), 2);
    assert_eq!(b.shape.len(), 2);
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a.data[i * k + p] * b.data[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    Tensor {
        data: out,
        shape: vec![m, n],
    }
}

fn bench_tensor_matmul(c: &mut Criterion) {
    let size = 128;
    let mut rng = rand::thread_rng();
    let a_data: Vec<f32> = (0..size * size).map(|_| rng.gen()).collect();
    let b_data: Vec<f32> = (0..size * size).map(|_| rng.gen()).collect();
    let a = Tensor::new(a_data, vec![size, size]);
    let b = Tensor::new(b_data, vec![size, size]);

    c.bench_function("tensor_matmul_naive", |bencher| {
        bencher.iter(|| {
            let res = tensor_matmul_naive(black_box(&a), black_box(&b));
            black_box(res);
        });
    });

    c.bench_function("tensor_matmul_new", |bencher| {
        bencher.iter(|| {
            let res = tensor_matmul(black_box(&a), black_box(&b));
            black_box(res);
        });
    });
}

criterion_group!(benches, bench_tensor_matmul);
criterion_main!(benches);
