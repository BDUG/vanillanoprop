use crate::autograd::Tensor;
use crate::data::to_matrix;
use crate::decoder_t::DecoderT;

pub fn greedy_decode(
    decoder: &DecoderT,
    enc_out: &Tensor,
    start_id: usize,
    end_id: usize,
    vocab_size: usize,
    max_len: usize,
) -> Vec<usize> {
    let mut seq = vec![start_id];
    for _ in 0..max_len {
        let tin = to_matrix(&seq, vocab_size);
        let logits = decoder.forward(&Tensor::from_matrix(tin), enc_out);
        let probs = Tensor::softmax(&logits);
        let last = probs.data.rows - 1;
        let mut best_tok = 0;
        let mut best_p = f32::NEG_INFINITY;
        for tok in 0..vocab_size {
            let p = probs.data.get(last, tok);
            if p > best_p {
                best_p = p;
                best_tok = tok;
            }
        }
        seq.push(best_tok);
        if best_tok == end_id {
            break;
        }
    }
    seq
}
