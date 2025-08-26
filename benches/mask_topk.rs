use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;

fn mask_topk_sort(data: &mut [f32], cols: usize, top_k: usize) {
    for row in data.chunks_mut(cols) {
        if top_k >= cols { continue; }
        let mut indices: Vec<usize> = (0..cols).collect();
        indices.sort_by(|&a, &b| {
            row[b].partial_cmp(&row[a]).unwrap()
        });
        for &idx in indices[top_k..].iter() {
            row[idx] = f32::NEG_INFINITY;
        }
    }
}

fn mask_topk_select(data: &mut [f32], cols: usize, top_k: usize) {
    for row in data.chunks_mut(cols) {
        if top_k >= cols { continue; }
        let mut indices: Vec<usize> = (0..cols).collect();
        indices.select_nth_unstable_by(top_k, |&a, &b| {
            row[b].partial_cmp(&row[a]).unwrap()
        });
        for &idx in indices[top_k..].iter() {
            row[idx] = f32::NEG_INFINITY;
        }
    }
}

fn bench_mask_topk(c: &mut Criterion) {
    let rows = 128;
    let cols = 64;
    let top_k = 16;
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..rows*cols).map(|_| rng.gen()).collect();

    c.bench_function("mask_topk_sort", |b| {
        b.iter(|| {
            let mut d = data.clone();
            mask_topk_sort(black_box(&mut d), cols, top_k);
        })
    });

    c.bench_function("mask_topk_select", |b| {
        b.iter(|| {
            let mut d = data.clone();
            mask_topk_select(black_box(&mut d), cols, top_k);
        })
    });
}

criterion_group!(benches, bench_mask_topk);
criterion_main!(benches);
