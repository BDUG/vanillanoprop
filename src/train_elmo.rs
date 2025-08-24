use crate::data::{load_pairs, to_matrix, Vocab};
use crate::math;
use crate::metrics::f1_score;
use crate::transformer_t::EncoderT;
use crate::weights::save_model;
use indicatif::ProgressBar;

pub fn run() {
    let pairs = load_pairs();
    let vocab = Vocab::build();
    let vocab_size = vocab.itos.len();

    // With embedding → model_dim separate
    let model_dim = 64;
    let mut encoder = EncoderT::new(6, vocab_size, model_dim, 128);
    let lr = 0.001;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;

    math::reset_matrix_ops();
    let epochs = 5;
    let pb = ProgressBar::new(epochs as u64);
    let mut best_f1 = f32::NEG_INFINITY;
    for epoch in 0..epochs {
        let mut last_loss = 0.0;
        let mut f1_sum = 0.0;
        let mut sample_cnt: f32 = 0.0;
        for (src, tgt) in &pairs {
            encoder.zero_grad();
            let x = to_matrix(src, vocab_size);
            let logits = encoder.forward_train(&x);
            let (loss, grad, preds) = math::softmax_cross_entropy(&logits, tgt, 0);
            last_loss = loss;

            // backprop + optimisation
            encoder.backward(&grad);
            encoder.adam_step(lr, beta1, beta2, eps);

            let f1 = f1_score(&preds, tgt);
            f1_sum += f1;
            sample_cnt += 1.0;
            println!("loss {loss:.4} f1 {f1:.4}");
        }
        let avg_f1 = f1_sum / if sample_cnt > 0.0 { sample_cnt } else { 1.0 };
        pb.set_message(format!("epoch {epoch} loss {last_loss:.4} f1 {avg_f1:.4}"));
        pb.inc(1);

        if avg_f1 > best_f1 {
            println!("Checkpoint saved at epoch {epoch}: avg F1 improved to {avg_f1:.4}");
            best_f1 = avg_f1;
            save_model("checkpoint.json", &encoder, None);
        }
    }
    pb.finish_with_message("training done");

    println!("Total matrix ops: {}", math::matrix_ops_count());

    save_model("model.json", &encoder, None);
}
