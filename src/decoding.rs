use crate::autograd::Tensor;
use crate::data::to_matrix;
use crate::transformer_t::DecoderT;

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

pub fn beam_search_decode(
    decoder: &DecoderT,
    enc_out: &Tensor,
    start_id: usize,
    end_id: usize,
    vocab_size: usize,
    max_len: usize,
    beam_size: usize,
) -> Vec<usize> {
    let mut beams: Vec<(Vec<usize>, f32)> = vec![(vec![start_id], 0.0)];
    for _ in 0..max_len {
        let mut candidates: Vec<(Vec<usize>, f32)> = Vec::new();
        for (seq, score) in beams.iter() {
            if *seq.last().unwrap() == end_id {
                candidates.push((seq.clone(), *score));
                continue;
            }
            let tin = to_matrix(seq, vocab_size);
            let logits = decoder.forward(&Tensor::from_matrix(tin), enc_out);
            let probs = Tensor::softmax(&logits);
            let last = probs.data.rows - 1;
            for tok in 0..probs.data.cols {
                let p = probs.data.get(last, tok).max(1e-9).ln();
                let mut new_seq = seq.clone();
                new_seq.push(tok);
                candidates.push((new_seq, score + p));
            }
        }
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        beams = candidates.into_iter().take(beam_size).collect();
        if beams.iter().any(|(seq, _)| *seq.last().unwrap() == end_id) {
            break;
        }
    }
    beams.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    beams[0].0.clone()
}
