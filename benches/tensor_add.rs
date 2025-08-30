use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use vanillanoprop::tensor::Tensor;

fn bench_tensor_add(c: &mut Criterion) {
    let size = 1_000_000; // 1 million elements
    let mut rng = rand::thread_rng();
    let a_data: Vec<f32> = (0..size).map(|_| rng.gen()).collect();
    let b_data: Vec<f32> = (0..size).map(|_| rng.gen()).collect();
    let a = Tensor::new(a_data, vec![size]);
    let b = Tensor::new(b_data, vec![size]);

    c.bench_function("tensor_add", |bencher| {
        bencher.iter(|| {
            let res = Tensor::add(black_box(&a), black_box(&b));
            black_box(res);
        });
    });
}

criterion_group!(benches, bench_tensor_add);
criterion_main!(benches);
