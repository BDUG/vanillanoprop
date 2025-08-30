use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use vanillanoprop::layers::linear::LinearT;
use vanillanoprop::tensor::Tensor;

fn quantized_linear_loop(x: &Tensor, w: &Tensor) -> Tensor {
    let (x_q, x_scale) = x.quantize();
    let (w_q, w_scale) = w.quantize();
    let rows = x.shape[0];
    let k = x.shape[1];
    let cols = w.shape[1];
    let mut out = vec![0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0i32;
            for p in 0..k {
                let a = x_q[i * k + p] as i32;
                let b = w_q[p * cols + j] as i32;
                sum += a * b;
            }
            out[i * cols + j] = sum as f32 / (x_scale * w_scale);
        }
    }
    Tensor::new(out, vec![rows, cols])
}

fn bench_linear(c: &mut Criterion) {
    let rows = 128;
    let in_dim = 256;
    let out_dim = 256;
    let mut rng = rand::thread_rng();
    let x_data: Vec<f32> = (0..rows * in_dim).map(|_| rng.gen()).collect();
    let x = Tensor::new(x_data, vec![rows, in_dim]);

    let mut layer = LinearT::new(in_dim, out_dim);
    let w = layer.w.clone();
    // warm up to cache quantized weights
    let _ = layer.quantized_matmul(&x);

    c.bench_function("quantized_linear_loop", |b| {
        b.iter(|| {
            let res = quantized_linear_loop(black_box(&x), black_box(&w));
            black_box(res);
        });
    });

    c.bench_function("quantized_linear_accel", |b| {
        b.iter(|| {
            let res = layer.quantized_matmul(black_box(&x));
            black_box(res);
        });
    });
}

criterion_group!(benches, bench_linear);
criterion_main!(benches);
