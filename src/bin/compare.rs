use indicatif::ProgressBar;
use vanillanoprop::data::{download_mnist, load_batches};
use vanillanoprop::math::{self, Matrix};
use vanillanoprop::metrics::f1_score;
use vanillanoprop::models::{DecoderT, EncoderT};
use vanillanoprop::optim::Adam;

fn to_matrix(seq: &[u8], vocab_size: usize) -> Matrix {
    let mut m = Matrix::zeros(seq.len(), vocab_size);
    for (i, &tok) in seq.iter().enumerate() {
        m.set(i, tok as usize, 1.0);
    }
    m
}

fn train_backprop(epochs: usize) -> (f32, usize) {
    let batches = load_batches(4);
    let vocab_size = 256;

    let model_dim = 64;
    let mut encoder = EncoderT::new(6, vocab_size, model_dim, 256);
    let mut decoder = DecoderT::new(6, vocab_size, model_dim, 256);

    let lr = 0.001;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;
    let weight_decay = 0.0;
    let mut adam = Adam::new(lr, beta1, beta2, eps, weight_decay);

    math::reset_matrix_ops();
    let pb = ProgressBar::new(epochs as u64);
    let mut best_f1 = f32::NEG_INFINITY;
    for epoch in 0..epochs {
        let mut last_loss = 0.0;
        let mut f1_sum = 0.0;
        let mut sample_cnt: f32 = 0.0;
        for batch in &batches {
            encoder.zero_grad();
            decoder.zero_grad();
            let mut batch_loss = 0.0f32;
            let mut batch_f1 = 0.0f32;
            for (src, tgt) in batch {
                let tgt = *tgt;
                let enc_x = to_matrix(src, vocab_size);
                let enc_out = encoder.forward_train(&enc_x);

                let dec_in = vec![tgt as u8];
                let dec_x = to_matrix(&dec_in, vocab_size);
                let logits = decoder.forward_train(&dec_x, &enc_out);

                let (loss, grad, preds) = math::softmax_cross_entropy(&logits, &[tgt], 0);
                batch_loss += loss;

                let grad_enc = decoder.backward(&grad);
                encoder.backward(&grad_enc);
                let f1 = f1_score(&preds, &[tgt]);
                batch_f1 += f1;
            }
            let bsz = batch.len() as f32;
            batch_loss /= bsz;
            let batch_f1_avg = batch_f1 / bsz;
            last_loss = batch_loss;
            f1_sum += batch_f1;
            sample_cnt += bsz;
            let mut params = encoder.parameters();
            {
                let dec_params = decoder.parameters();
                params.extend(dec_params);
            }
            adam.step(&mut params);
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
    let ops = math::matrix_ops_count();
    (best_f1, ops)
}

fn train_noprop(epochs: usize) -> (f32, usize) {
    let batches = load_batches(4);
    let vocab_size = 256;

    let model_dim = 64;
    let mut encoder = EncoderT::new(6, vocab_size, model_dim, 256);
    let lr = 0.001;

    math::reset_matrix_ops();
    let pb = ProgressBar::new(epochs as u64);
    let mut best_f1 = f32::NEG_INFINITY;
    for epoch in 0..epochs {
        let mut last_loss = 0.0;
        let mut f1_sum = 0.0;
        let mut sample_cnt: f32 = 0.0;
        for batch in &batches {
            let mut batch_loss = 0.0f32;
            let mut batch_f1 = 0.0f32;
            for (src, tgt) in batch {
                let tgt = *tgt;
                let len = 1usize;
                let x = to_matrix(&src[..len], vocab_size);
                let enc_out = encoder.forward_local(&x);

                let mut noisy = encoder.forward(&to_matrix(&[tgt as u8], vocab_size));
                for v in &mut noisy.data.data {
                    *v += (rand::random::<f32>() - 0.5) * 0.1;
                }

                let mut delta = Matrix::zeros(enc_out.rows, enc_out.cols);
                let mut loss = 0.0f32;
                for i in 0..len * model_dim {
                    let d = enc_out.data[i] - noisy.data.data[i];
                    loss += d * d;
                    delta.data[i] = 2.0 * d;
                }
                let n = (len * model_dim) as f32;
                if n > 0.0 {
                    loss /= n;
                    for v in delta.data.iter_mut() {
                        *v /= n;
                    }
                }

                batch_loss += loss;
                encoder.fa_update(&delta, lr);
                let src_slice: Vec<usize> = src[..len].iter().map(|&v| v as usize).collect();
                let f1 = f1_score(&src_slice, &[tgt]);
                batch_f1 += f1;
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
    let ops = math::matrix_ops_count();
    (best_f1, ops)
}

fn main() {
    download_mnist();
    let epochs = 10;
    println!("Running backpropagation for {epochs} epochs...");
    let (bp_f1, bp_ops) = train_backprop(epochs);
    println!("Running noprop for {epochs} epochs...");
    let (np_f1, np_ops) = train_noprop(epochs);
    println!("\nComparison after {epochs} epochs:");
    println!("Backprop -> Best F1: {bp_f1:.4}, Matrix Ops: {bp_ops}");
    println!("Noprop   -> Best F1: {np_f1:.4}, Matrix Ops: {np_ops}");
}
