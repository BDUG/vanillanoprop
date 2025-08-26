use indicatif::ProgressBar;
use rand::Rng;
use vanillanoprop::rng::rng_from_env;

use vanillanoprop::data::{download_mnist, load_batches};
use vanillanoprop::math;
use vanillanoprop::memory;
use vanillanoprop::metrics::f1_score;
use vanillanoprop::models::SimpleCNN;

// Train a SimpleCNN with standard backpropagation using a basic SGD loop.
fn train_backprop(epochs: usize) -> (f32, usize, usize, u64) {
    let batches = load_batches(4);
    let mut cnn = SimpleCNN::new(10);

    let lr = 0.01f32;

    math::reset_matrix_ops();
    let start_mem = memory::peak_memory_bytes();
    let pb = ProgressBar::new(epochs as u64);
    let mut best_f1 = f32::NEG_INFINITY;

    for epoch in 0..epochs {
        let mut last_loss = 0.0f32;
        let mut f1_sum = 0.0f32;
        let mut sample_cnt = 0.0f32;

        for batch in &batches {
            let mut batch_loss = 0.0f32;
            let mut batch_f1 = 0.0f32;

            for (img, tgt) in batch {
                let (feat, logits) = cnn.forward(img);

                // Softmax cross entropy
                let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                let mut probs = vec![0f32; logits.len()];
                for (i, &v) in logits.iter().enumerate() {
                    let e = (v - max).exp();
                    probs[i] = e;
                    exp_sum += e;
                }
                for p in &mut probs {
                    *p /= exp_sum;
                }

                let loss = -probs[*tgt as usize].ln();
                batch_loss += loss;

                let mut grad_logits = probs.clone();
                grad_logits[*tgt as usize] -= 1.0;

                // Update weights
                let (fc, bias) = cnn.parameters_mut();
                let rows = fc.rows;
                let cols = fc.cols;
                for c in 0..cols {
                    let g = grad_logits[c];
                    bias[c] -= lr * g;
                    for r in 0..rows {
                        let val = fc.get(r, c) - lr * g * feat[r];
                        fc.set(r, c, val);
                    }
                }

                // Metrics
                let mut best = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for (i, &v) in probs.iter().enumerate() {
                    if v > best_val {
                        best_val = v;
                        best = i;
                    }
                }
                batch_f1 += f1_score(&[best], &[*tgt as usize]);
            }

            let bsz = batch.len() as f32;
            batch_loss /= bsz;
            let batch_f1_avg = batch_f1 / bsz;
            last_loss = batch_loss;
            f1_sum += batch_f1;
            sample_cnt += bsz;
            println!("backprop epoch {epoch} batch loss {batch_loss:.4} f1 {batch_f1_avg:.4}");
        }

        let avg_f1 = f1_sum / if sample_cnt > 0.0 { sample_cnt } else { 1.0 };
        pb.set_message(format!("epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}"));
        pb.inc(1);
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
    let batches = load_batches(4);
    let mut cnn = SimpleCNN::new(10);
    let mut rng = rng_from_env();

    let lr = 0.01f32;

    math::reset_matrix_ops();
    let start_mem = memory::peak_memory_bytes();
    let pb = ProgressBar::new(epochs as u64);
    let mut best_f1 = f32::NEG_INFINITY;

    for epoch in 0..epochs {
        let mut last_loss = 0.0f32;
        let mut f1_sum = 0.0f32;
        let mut sample_cnt = 0.0f32;

        for batch in &batches {
            let mut batch_loss = 0.0f32;
            let mut batch_f1 = 0.0f32;

            for (img, tgt) in batch {
                let (feat, logits) = cnn.forward(img);

                // Create noisy one-hot target
                let mut target = vec![0f32; logits.len()];
                target[*tgt as usize] = 1.0;
                for v in &mut target {
                    *v += (rng.gen::<f32>() - 0.5) * 0.1;
                }

                let mut delta = vec![0f32; logits.len()];
                let mut loss = 0.0f32;
                for i in 0..logits.len() {
                    let d = logits[i] - target[i];
                    loss += d * d;
                    delta[i] = 2.0 * d / logits.len() as f32;
                }
                loss /= logits.len() as f32;
                batch_loss += loss;

                // Local weight update
                let (fc, bias) = cnn.parameters_mut();
                let rows = fc.rows;
                let cols = fc.cols;
                for c in 0..cols {
                    let g = delta[c];
                    bias[c] -= lr * g;
                    for r in 0..rows {
                        let val = fc.get(r, c) - lr * g * feat[r];
                        fc.set(r, c, val);
                    }
                }

                // Metrics
                let mut best = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for (i, &v) in logits.iter().enumerate() {
                    if v > best_val {
                        best_val = v;
                        best = i;
                    }
                }
                batch_f1 += f1_score(&[best], &[*tgt as usize]);
            }

            let bsz = batch.len() as f32;
            batch_loss /= bsz;
            let batch_f1_avg = batch_f1 / bsz;
            last_loss = batch_loss;
            f1_sum += batch_f1;
            sample_cnt += bsz;
            println!("noprop epoch {epoch} batch loss {batch_loss:.4} f1 {batch_f1_avg:.4}");
        }

        let avg_f1 = f1_sum / if sample_cnt > 0.0 { sample_cnt } else { 1.0 };
        pb.set_message(format!("epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}"));
        pb.inc(1);
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
    download_mnist();
    let epochs = 5;
    println!("Running backpropagation for {epochs} epochs...");
    let (bp_f1, bp_add, bp_mul, bp_mem) = train_backprop(epochs);
    println!("Running noprop for {epochs} epochs...");
    let (np_f1, np_add, np_mul, np_mem) = train_noprop(epochs);
    println!("\nComparison after {epochs} epochs:");
    println!(
        "Backprop -> Best F1: {bp_f1:.4}, Adds: {bp_add}, Muls: {bp_mul}, Peak Mem: {:.2} MB",
        bp_mem as f64 / (1024.0 * 1024.0)
    );
    println!(
        "Noprop   -> Best F1: {np_f1:.4}, Adds: {np_add}, Muls: {np_mul}, Peak Mem: {:.2} MB",
        np_mem as f64 / (1024.0 * 1024.0)
    );
}

