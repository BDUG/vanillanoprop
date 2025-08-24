use crate::data::{load_pairs, to_matrix, Vocab, END, START};
use crate::decoding::beam_search_decode;
use crate::math;
use crate::models::{DecoderT, EncoderT};
use crate::weights::load_model;
use rand::Rng;

pub fn run() {
    // pick a random image from the MNIST training pairs
    let pairs = load_pairs();
    let mut rng = rand::thread_rng();
    let idx = rng.gen_range(0..pairs.len());
    let (src, tgt) = &pairs[idx];

    let vocab = Vocab::build();
    let vocab_size = vocab.itos.len();
    let start_id = *vocab.stoi.get(START).unwrap();
    let end_id = *vocab.stoi.get(END).unwrap();

    let model_dim = 64;
    let mut encoder = EncoderT::new(6, vocab_size, model_dim, 256);
    let mut decoder = DecoderT::new(6, vocab_size, model_dim, 256);

    load_model("model.json", &mut encoder, &mut decoder);

    math::reset_matrix_ops();
    let enc_x = to_matrix(src, vocab_size);
    let enc_out = encoder.forward(&enc_x);

    // use beam search for better predictions
    let out_ids = beam_search_decode(&decoder, &enc_out, start_id, end_id, vocab_size, 50, 3);
    let pred_id = out_ids
        .into_iter()
        .find(|&id| id != start_id && id != end_id);
    let prediction = pred_id
        .map(|id| vocab.itos[id].clone())
        .unwrap_or_else(|| "[no prediction]".to_string());
    let actual = tgt
        .first()
        .map(|&id| vocab.itos[id].clone())
        .unwrap_or_else(|| "[unknown]".to_string());
    println!(
        "{{\"actual\":\"{}\", \"prediction\":\"{}\"}}",
        actual, prediction
    );
    println!("Total matrix ops: {}", math::matrix_ops_count());
}
