use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use vanillanoprop::math::{self, Matrix};

// Baseline implementation copied from previous version
fn softmax_cross_entropy_old(
    logits: &Matrix,
    targets: &[usize],
    row_offset: usize,
) -> (f32, Matrix, Vec<usize>) {
    let probs = logits.softmax();
    let mut grad = probs.clone();
    let mut loss = 0.0f32;
    let mut preds = Vec::new();
    let mut cnt = 0.0f32;

    for (i, &tok) in targets.iter().enumerate() {
        let row = i + row_offset;
        if row >= logits.rows {
            break;
        }

        let mut best_tok = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for t in 0..logits.cols {
            let p = probs.get(row, t);
            if p > best_val {
                best_val = p;
                best_tok = t;
            }
        }

        let p = probs.get(row, tok);
        loss += -(p + 1e-9).ln();
        grad.set(row, tok, grad.get(row, tok) - 1.0);
        preds.push(best_tok);
        cnt += 1.0;
    }

    if cnt > 0.0 {
        loss /= cnt;
        for v in grad.data.iter_mut() {
            *v /= cnt;
        }
    }

    (loss, grad, preds)
}

fn bench_softmax_ce(c: &mut Criterion) {
    // Use reasonably large matrices so memory usage differences are visible.
    let rows = 512;
    let cols = 512;
    let mut rng = rand::thread_rng();
    let logits_vec: Vec<f32> = (0..rows * cols)
        .map(|_| rng.gen_range(-10.0..10.0))
        .collect();
    let logits = Matrix::from_vec(rows, cols, logits_vec);
    let targets: Vec<usize> = (0..rows).map(|_| rng.gen_range(0..cols)).collect();

    c.bench_function("softmax_ce_old", |b| {
        b.iter(|| {
            let res = softmax_cross_entropy_old(black_box(&logits), black_box(&targets), 0);
            black_box(res);
        });
    });

    c.bench_function("softmax_ce_new", |b| {
        b.iter(|| {
            let res = math::softmax_cross_entropy(black_box(&logits), black_box(&targets), 0);
            black_box(res);
        });
    });

    let bytes_per_matrix = rows * cols * std::mem::size_of::<f32>();
    let old_mem = bytes_per_matrix * 2; // probs + grad
    let new_mem = bytes_per_matrix; // grad only
    println!(
        "theoretical extra memory -> old: {} bytes, new: {} bytes",
        old_mem, new_mem
    );
}

criterion_group!(benches, bench_softmax_ce);
criterion_main!(benches);
