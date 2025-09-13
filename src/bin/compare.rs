use vanillanoprop::util::progress::ProgressBar;
use rand::Rng;
use vanillanoprop::rng::rng_from_env;

use vanillanoprop::data::{download_mnist, DataLoader, Mnist};
use vanillanoprop::math::{self, Matrix};
use vanillanoprop::memory;
use vanillanoprop::metrics::f1_score;
use vanillanoprop::models::SimpleCNN;

mod common;

// Train a SimpleCNN with standard backpropagation using a basic SGD loop.
fn train_backprop(epochs: usize) -> (f32, usize, usize, u64) {
    let mut cnn = SimpleCNN::new(10);

    let lr = 0.01f32;

    math::reset_matrix_ops();
    let start_mem = memory::peak_memory_bytes();
    let mut pb = ProgressBar::new(epochs as u64);
    let mut best_f1 = f32::NEG_INFINITY;

    let mut loader = DataLoader::<Mnist>::new(4, false, None);
    for epoch in 0..epochs {
        let mut last_loss = 0.0f32;
        let mut f1_sum = 0.0f32;
        let mut sample_cnt = 0.0f32;
        loader.reset(true);
        for batch in loader.by_ref() {
            let bsz = batch.len();
            let mut images = Vec::with_capacity(bsz);
            let mut targets = Vec::with_capacity(bsz);
            for (img, tgt) in batch.iter() {
                images.push(img.clone());
                targets.push(*tgt as usize);
            }

            let (feat_m, logits) = cnn.forward_batch(&images);
            let (batch_loss, grad_m, preds) =
                math::softmax_cross_entropy(&logits, &targets, 0);

            let (fc, bias) = cnn.parameters_mut();
            let grad_fc = Matrix::matmul(&feat_m.transpose(), &grad_m);
            for (w, g) in fc.data.iter_mut().zip(grad_fc.data.iter()) {
                *w -= lr * g;
            }
            let mut grad_bias = vec![0.0f32; grad_m.cols];
            for r in 0..grad_m.rows {
                for c in 0..grad_m.cols {
                    grad_bias[c] += grad_m.get(r, c);
                }
            }
            for (b, g) in bias.iter_mut().zip(grad_bias.iter()) {
                *b -= lr * g;
            }

            let batch_f1 = f1_score(&preds, &targets);
            last_loss = batch_loss;
            f1_sum += batch_f1;
            sample_cnt += 1.0;
        }

        let avg_f1 = f1_sum / if sample_cnt > 0.0 { sample_cnt } else { 1.0 };
        pb.set_message(format!("epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}"));
        pb.inc(1);
        vanillanoprop::info!(
            "backprop epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}"
        );
        if avg_f1 > best_f1 {
            best_f1 = avg_f1;
        }
    }
    pb.finish_with_message("backprop done");
    let add_ops = math::add_ops_count();
    let mul_ops = math::mul_ops_count();
    let mem_used = memory::peak_memory_bytes() - start_mem;
    (best_f1, add_ops, mul_ops, mem_used)
}

// Train a SimpleCNN using a NoProp-style local update with noisy targets.
fn train_noprop(epochs: usize) -> (f32, usize, usize, u64) {
    let mut cnn = SimpleCNN::new(10);
    let mut rng = rng_from_env();

    let lr = 0.01f32;

    math::reset_matrix_ops();
    let start_mem = memory::peak_memory_bytes();
    let mut pb = ProgressBar::new(epochs as u64);
    let mut best_f1 = f32::NEG_INFINITY;

    let mut loader = DataLoader::<Mnist>::new(4, false, None);
    for epoch in 0..epochs {
        let mut last_loss = 0.0f32;
        let mut f1_sum = 0.0f32;
        let mut sample_cnt = 0.0f32;
        loader.reset(true);
        for batch in loader.by_ref() {
            let bsz = batch.len();
            let mut images = Vec::with_capacity(bsz);
            let mut targets = Vec::with_capacity(bsz);
            for (img, tgt) in batch.iter() {
                images.push(img.clone());
                targets.push(*tgt as usize);
            }

            let (feat_m, logits) = cnn.forward_batch(&images);
            let mut delta = Matrix::zeros(bsz, logits.cols);
            let mut batch_loss = 0.0f32;
            for i in 0..bsz {
                let mut target = vec![0f32; logits.cols];
                target[targets[i]] = 1.0;
                for v in &mut target {
                    *v += (rng.gen::<f32>() - 0.5) * 0.1;
                }
                for c in 0..logits.cols {
                    let l = logits.get(i, c);
                    let d = l - target[c];
                    batch_loss += d * d / logits.cols as f32;
                    delta.set(i, c, 2.0 * d / logits.cols as f32);
                }
            }
            batch_loss /= bsz as f32;

            let (fc, bias) = cnn.parameters_mut();
            let grad_fc = Matrix::matmul(&feat_m.transpose(), &delta);
            for (w, g) in fc.data.iter_mut().zip(grad_fc.data.iter()) {
                *w -= lr * g;
            }
            let mut grad_bias = vec![0.0f32; delta.cols];
            for r in 0..delta.rows {
                for c in 0..delta.cols {
                    grad_bias[c] += delta.get(r, c);
                }
            }
            for (b, g) in bias.iter_mut().zip(grad_bias.iter()) {
                *b -= lr * g;
            }

            let mut preds = Vec::with_capacity(bsz);
            for i in 0..bsz {
                let row = &logits.data[i * logits.cols..(i + 1) * logits.cols];
                preds.push(math::argmax(row));
            }
            let batch_f1 = f1_score(&preds, &targets);
            last_loss = batch_loss;
            f1_sum += batch_f1;
            sample_cnt += 1.0;
        }

        let avg_f1 = f1_sum / if sample_cnt > 0.0 { sample_cnt } else { 1.0 };
        pb.set_message(format!("epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}"));
        pb.inc(1);
        vanillanoprop::info!("noprop epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}");
        if avg_f1 > best_f1 {
            best_f1 = avg_f1;
        }
    }
    pb.finish_with_message("noprop done");
    let add_ops = math::add_ops_count();
    let mul_ops = math::mul_ops_count();
    let mem_used = memory::peak_memory_bytes() - start_mem;
    (best_f1, add_ops, mul_ops, mem_used)
}

fn main() {
    let _ = common::init_logging();
    download_mnist();
    let epochs = 5;
    vanillanoprop::info!("Running backpropagation for {epochs} epochs...");
    let (bp_f1, bp_add, bp_mul, bp_mem) = train_backprop(epochs);
    vanillanoprop::info!("Running noprop for {epochs} epochs...");
    let (np_f1, np_add, np_mul, np_mem) = train_noprop(epochs);
    vanillanoprop::info!("\nComparison after {epochs} epochs:");
    vanillanoprop::info!(
        "Backprop -> Best F1: {bp_f1:.4}, Adds: {bp_add}, Muls: {bp_mul}, Peak Mem: {:.2} MB",
        bp_mem as f64 / (1024.0 * 1024.0)
    );
    vanillanoprop::info!(
        "Noprop   -> Best F1: {np_f1:.4}, Adds: {np_add}, Muls: {np_mul}, Peak Mem: {:.2} MB",
        np_mem as f64 / (1024.0 * 1024.0)
    );
}
