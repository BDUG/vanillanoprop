use crate::data::{to_matrix, Vocab, START, END};
use crate::transformer_t::{DecoderT, EncoderT};
use crate::decoding::greedy_decode;
use crate::weights::load_model;

pub fn run(input: &str) {
    let vocab = Vocab::build();
    let vocab_size = vocab.itos.len();
    let start_id = *vocab.stoi.get(START).unwrap();
    let end_id = *vocab.stoi.get(END).unwrap();

    let model_dim = 64;
    let mut encoder = EncoderT::new(6, vocab_size, model_dim, 1, 256);
    let mut decoder = DecoderT::new(6, vocab_size, model_dim, 1, 256);

    load_model("model.json", &mut encoder, &mut decoder);

    let src = vocab.encode(input);
    let enc_x = to_matrix(&src, vocab_size);
    let enc_out = encoder.forward(&enc_x);

    let out_ids = greedy_decode(&decoder, &enc_out, start_id, end_id, vocab_size, 50);
    let mut filtered = Vec::new();
    for &id in out_ids.iter() {
        if id == start_id {
            continue;
        }
        if id == end_id {
            break;
        }
        filtered.push(id);
    }
    let translation = vocab.decode(&filtered);
    println!("{{\"input\":\"{}\", \"translation\":\"{}\"}}", input, translation);
}
