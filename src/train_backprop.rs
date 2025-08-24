use crate::data::{load_batches, to_matrix, Vocab, START};
use crate::math;
use crate::metrics::f1_score;
use crate::optim::{Adam, SGD};
use crate::models::{DecoderT, EncoderT};
use crate::weights::save_model;
use indicatif::ProgressBar;

// Tensor Backprop Training (simplified Adam hook)
// now using Embedding => model_dim independent of vocab_size
pub fn run(_opt: &str) {
    let batches = load_batches(4);
    let vocab = Vocab::build();
    let vocab_size = vocab.itos.len();

    let model_dim = 64;
    let mut encoder = EncoderT::new(6, vocab_size, model_dim, 256);
    let mut decoder = DecoderT::new(6, vocab_size, model_dim, 256);

    let lr = 0.001;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;
    let weight_decay = 0.0;
    let mut adam = Adam::new(lr, beta1, beta2, eps, weight_decay);
    let mut sgd = SGD::new(lr, weight_decay);
    let start_id = *vocab.stoi.get(START).unwrap();

    math::reset_matrix_ops();
    let epochs = 50;
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
                // Encode source sentence
                let enc_x = to_matrix(src, vocab_size);
                let enc_out = encoder.forward_train(&enc_x);

                // Decoder input uses teacher forcing (START + target[:-1])
                let mut dec_in = vec![start_id];
                dec_in.extend_from_slice(tgt);
                let dec_x = to_matrix(&dec_in, vocab_size);
                let logits = decoder.forward_train(&dec_x, &enc_out);

                let (loss, grad, preds) =
                    math::softmax_cross_entropy(&logits, tgt, 1);
                batch_loss += loss;

                // Backward through decoder and encoder
                let grad_enc = decoder.backward(&grad);
                encoder.backward(&grad_enc);
                let f1 = f1_score(&preds, tgt);
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
            if _opt == "sgd" {
                sgd.step(&mut params);
            } else {
                adam.step(&mut params);
            }
            println!("loss {batch_loss:.4} f1 {batch_f1_avg:.4}");
        }
        let avg_f1 = f1_sum / if sample_cnt > 0.0 { sample_cnt } else { 1.0 };
        pb.set_message(format!("epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}"));
        pb.inc(1);

        if avg_f1 > best_f1 {
            println!("Checkpoint saved at epoch {epoch}: avg F1 improved to {avg_f1:.4}");
            best_f1 = avg_f1;
            save_model("checkpoint.json", &mut encoder, Some(&mut decoder));
        }
    }
    pb.finish_with_message("training done");

    println!("Total matrix ops: {}", math::matrix_ops_count());

    // Save trained weights
    save_model("model.json", &mut encoder, Some(&mut decoder));
}
