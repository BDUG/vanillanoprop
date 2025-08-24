use crate::data::{load_mnist_pairs, to_matrix, Vocab, END, START};
use crate::decoding::beam_search_decode;
use crate::transformer_t::{DecoderT, EncoderT};
use crate::weights::load_model;
use rand::Rng;

pub fn run() {
    // pick a random image from the MNIST training pairs
    let pairs = load_mnist_pairs();
    let mut rng = rand::thread_rng();
    let idx = rng.gen_range(0..pairs.len());
    let (src, tgt) = &pairs[idx];

    let vocab = Vocab::build_mnist();
    let vocab_size = vocab.itos.len();
    let start_id = *vocab.stoi.get(START).unwrap();
    let end_id = *vocab.stoi.get(END).unwrap();

    let model_dim = 64;
    let mut encoder = EncoderT::new(6, vocab_size, model_dim, 1, 256);
    let mut decoder = DecoderT::new(6, vocab_size, model_dim, 1, 256);

    load_model("model.json", &mut encoder, &mut decoder);

    let enc_x = to_matrix(src, vocab_size);
    let enc_out = encoder.forward(&enc_x);

    // use beam search for better predictions
    let out_ids = beam_search_decode(&decoder, &enc_out, start_id, end_id, vocab_size, 50, 3);
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
    let prediction = if filtered.is_empty() {
        "[no prediction]".to_string()
    } else {
        vocab.decode(&filtered)
    };
    let actual = if tgt.len() > 1 {
        vocab.decode(&tgt[..tgt.len() - 1])
    } else {
        "[unknown]".to_string()
    };
    println!(
        "{{\"actual\":\"{}\", \"prediction\":\"{}\"}}",
        actual, prediction
    );
}

