use std::sync::atomic::{AtomicUsize, Ordering};

static MATRIX_OPS: AtomicUsize = AtomicUsize::new(0);

pub fn reset_matrix_ops() {
    MATRIX_OPS.store(0, Ordering::SeqCst);
}

pub fn matrix_ops_count() -> usize {
    MATRIX_OPS.load(Ordering::SeqCst)
}

pub(crate) fn inc_ops() {
    MATRIX_OPS.fetch_add(1, Ordering::SeqCst);
}

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn zeros(r: usize, c: usize) -> Self {
        Matrix {
            rows: r,
            cols: c,
            data: vec![0.0; r * c],
        }
    }

    pub fn from_vec(r: usize, c: usize, v: Vec<f32>) -> Self {
        assert_eq!(v.len(), r * c);
        Matrix {
            rows: r,
            cols: c,
            data: v,
        }
    }

    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        self.data[r * self.cols + c] = v;
    }

    pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
        inc_ops();
        assert_eq!(a.cols, b.rows);
        let mut out = vec![0.0; a.rows * b.cols];
        for i in 0..a.rows {
            let a_row = &a.data[i * a.cols..(i + 1) * a.cols];
            for k in 0..a.cols {
                let a_val = a_row[k];
                let b_row = &b.data[k * b.cols..(k + 1) * b.cols];
                for j in 0..b.cols {
                    out[i * b.cols + j] += a_val * b_row[j];
                }
            }
        }
        Matrix::from_vec(a.rows, b.cols, out)
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        inc_ops();
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut v = vec![0.0; self.data.len()];
        for i in 0..v.len() {
            v[i] = self.data[i] + other.data[i];
        }
        Matrix::from_vec(self.rows, self.cols, v)
    }

    pub fn transpose(&self) -> Matrix {
        inc_ops();
        let mut v = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                v[j * self.rows + i] = self.get(i, j);
            }
        }
        Matrix::from_vec(self.cols, self.rows, v)
    }

    pub fn softmax(&self) -> Matrix {
        inc_ops();
        let mut v = vec![0.0; self.data.len()];
        for r in 0..self.rows {
            // stabilisiert gegen Overflow:
            let row_start = r * self.cols;
            let row_slice = &self.data[row_start..row_start + self.cols];
            let max = row_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for c in 0..self.cols {
                let e = (self.get(r, c) - max).exp();
                v[row_start + c] = e;
                sum += e;
            }
            for c in 0..self.cols {
                v[row_start + c] /= sum;
            }
        }
        Matrix::from_vec(self.rows, self.cols, v)
    }
}

/// Compute a numerically stable softmax followed by cross-entropy loss.
///
/// The function returns the average loss, the gradient with respect to the
/// logits and the predicted token ids (argmax of the probabilities) for the
/// processed rows.  `targets` provides the expected token ids for each row and
/// `row_offset` allows skipping rows at the top of the `logits` matrix (useful
/// for decoder training where the first row corresponds to the start token).
pub fn softmax_cross_entropy(
    logits: &Matrix,
    targets: &[usize],
    row_offset: usize,
) -> (f32, Matrix, Vec<usize>) {
    // compute softmax probabilities
    let probs = logits.softmax();
    let mut grad = probs.clone();
    let mut loss = 0.0f32;
    let mut preds = Vec::new();
    let mut cnt = 0.0f32;

    for (i, &tok) in targets.iter().enumerate() {
        let row = i + row_offset;
        if row >= logits.rows {
            break;
        }

        // determine prediction via argmax
        let mut best_tok = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for t in 0..logits.cols {
            let p = probs.get(row, t);
            if p > best_val {
                best_val = p;
                best_tok = t;
            }
        }

        let p = probs.get(row, tok);
        loss += -(p + 1e-9).ln();
        grad.set(row, tok, grad.get(row, tok) - 1.0);
        preds.push(best_tok);
        cnt += 1.0;
    }

    if cnt > 0.0 {
        loss /= cnt;
        for v in grad.data.iter_mut() {
            *v /= cnt;
        }
    }

    (loss, grad, preds)
}
