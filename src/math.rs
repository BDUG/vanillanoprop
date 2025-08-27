use crate::device::{Cpu, Device};
use std::sync::atomic::{AtomicUsize, Ordering};

// Separate counters for addition and multiplication operations.
static ADD_OPS: AtomicUsize = AtomicUsize::new(0);
static MUL_OPS: AtomicUsize = AtomicUsize::new(0);

/// Reset the operation counters.
pub fn reset_matrix_ops() {
    ADD_OPS.store(0, Ordering::SeqCst);
    MUL_OPS.store(0, Ordering::SeqCst);
}

/// Return the total number of tracked operations (adds + muls).
pub fn matrix_ops_count() -> usize {
    ADD_OPS.load(Ordering::SeqCst) + MUL_OPS.load(Ordering::SeqCst)
}

/// Return the number of addition operations performed.
pub fn add_ops_count() -> usize {
    ADD_OPS.load(Ordering::SeqCst)
}

/// Return the number of multiplication operations performed.
pub fn mul_ops_count() -> usize {
    MUL_OPS.load(Ordering::SeqCst)
}

pub(crate) fn inc_add_ops_by(n: usize) {
    ADD_OPS.fetch_add(n, Ordering::SeqCst);
}

pub(crate) fn inc_mul_ops_by(n: usize) {
    MUL_OPS.fetch_add(n, Ordering::SeqCst);
}

/// Legacy helper for sites that tracked a combined count. The value is split
/// evenly between addition and multiplication counters.
pub(crate) fn inc_ops_by(n: usize) {
    let half = n / 2;
    inc_add_ops_by(half);
    inc_mul_ops_by(n - half);
}

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

/// CPU implementation of matrix multiplication. This is used by the default
/// [`Cpu`](crate::device::Cpu) device but exposed for other backends to reuse.
pub(crate) fn matmul_cpu(a: &Matrix, b: &Matrix) -> Matrix {
    // Each output element requires a.cols multiplications and additions
    let muls = a.rows * a.cols * b.cols;
    let adds = muls;
    inc_mul_ops_by(muls);
    inc_add_ops_by(adds);
    assert_eq!(a.cols, b.rows);

    const BLOCK: usize = 32;
    const PAR_THRESHOLD: usize = 128 * 128; // Use rayon when matrix is reasonably large

    let m = a.rows;
    let n = b.cols;
    let k_dim = a.cols;
    let mut out = vec![0.0; m * n];

    if m * n > PAR_THRESHOLD {
        use rayon::prelude::*;
        out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
            let a_row = &a.data[i * k_dim..(i + 1) * k_dim];
            for kk in (0..k_dim).step_by(BLOCK) {
                let k_end = (kk + BLOCK).min(k_dim);
                for k_idx in kk..k_end {
                    let a_val = a_row[k_idx];
                    let b_row = &b.data[k_idx * n..(k_idx + 1) * n];
                    for jj in (0..n).step_by(BLOCK) {
                        let j_end = (jj + BLOCK).min(n);
                        for j in jj..j_end {
                            out_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        });
    } else {
        for i in 0..m {
            let a_row = &a.data[i * k_dim..(i + 1) * k_dim];
            let out_row = &mut out[i * n..(i + 1) * n];
            for kk in (0..k_dim).step_by(BLOCK) {
                let k_end = (kk + BLOCK).min(k_dim);
                for k_idx in kk..k_end {
                    let a_val = a_row[k_idx];
                    let b_row = &b.data[k_idx * n..(k_idx + 1) * n];
                    for jj in (0..n).step_by(BLOCK) {
                        let j_end = (jj + BLOCK).min(n);
                        for j in jj..j_end {
                            out_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
    }

    Matrix::from_vec(m, n, out)
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

    /// Multiply `a` and `b` using the default [`Cpu`] device.
    pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
        Cpu.matmul(a, b)
    }

    /// Multiply `a` and `b` using the provided device implementation.
    pub fn matmul_with<D: Device>(a: &Matrix, b: &Matrix, device: &D) -> Matrix {
        device.matmul(a, b)
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        // One addition per element
        inc_add_ops_by(self.data.len());
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut v = Vec::with_capacity(self.data.len());
        for (&a, &b) in self.data.iter().zip(other.data.iter()) {
            v.push(a + b);
        }
        Matrix::from_vec(self.rows, self.cols, v)
    }

    pub fn transpose(&self) -> Matrix {
        let mut v = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                v[j * self.rows + i] = self.get(i, j);
            }
        }
        Matrix::from_vec(self.cols, self.rows, v)
    }

    pub fn softmax(&self) -> Matrix {
        // One addition per element when accumulating the sum
        inc_add_ops_by(self.rows * self.cols);
        let mut v = vec![0.0; self.data.len()];
        for (out_row, row_slice) in v.chunks_mut(self.cols).zip(self.data.chunks(self.cols)) {
            // stabilizes against overflow
            let max = row_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            // compute exponentials directly into the output buffer while accumulating the sum
            let mut sum = 0.0f32;
            for (out, &x) in out_row.iter_mut().zip(row_slice.iter()) {
                let e = (x - max).exp();
                *out = e;
                sum += e;
            }
            // normalize in-place
            for out in out_row.iter_mut() {
                *out /= sum;
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
    let mut grad = Matrix::zeros(logits.rows, logits.cols);
    let mut loss = 0.0f32;
    let mut preds = Vec::new();
    let mut cnt = 0.0f32;

    for (i, &tok) in targets.iter().enumerate() {
        let row = i + row_offset;
        if row >= logits.rows {
            break;
        }

        // Slice for the current row to avoid repeated index calculations
        let row_slice = &logits.data[row * logits.cols..(row + 1) * logits.cols];
        let grad_row = &mut grad.data[row * logits.cols..(row + 1) * logits.cols];

        // First pass: determine maximum logit for numerical stability and
        // simultaneously compute the argmax for predictions.
        let mut max_val = f32::NEG_INFINITY;
        let mut best_tok = 0usize;
        for (t, &v) in row_slice.iter().enumerate() {
            if v > max_val {
                max_val = v;
                best_tok = t;
            }
        }

        // Second pass: compute exponentials and their sum while storing them
        // directly into the gradient buffer.
        let mut sum = 0.0f32;
        for (g, &v) in grad_row.iter_mut().zip(row_slice.iter()) {
            let e = (v - max_val).exp();
            *g = e;
            sum += e;
        }

        // Normalize to obtain probabilities, accumulate loss and finalize
        // gradient in-place.
        let mut target_prob = 0.0f32;
        for (t, g) in grad_row.iter_mut().enumerate() {
            *g /= sum;
            if t == tok {
                target_prob = *g;
                *g -= 1.0;
            }
        }
        loss += -(target_prob + 1e-9).ln();
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

/// Return the index of the maximum value in `v`.
pub fn argmax(v: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &val) in v.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }
    best_idx
}

/// Mean squared error loss between `pred` and `target` matrices.
/// Returns the average loss and the gradient with respect to `pred`.
pub fn mse_loss(pred: &Matrix, target: &Matrix) -> (f32, Matrix) {
    assert_eq!(pred.rows, target.rows);
    assert_eq!(pred.cols, target.cols);
    let mut grad = Matrix::zeros(pred.rows, pred.cols);
    let mut loss = 0.0f32;
    for (i, (&p, &t)) in pred.data.iter().zip(target.data.iter()).enumerate() {
        let diff = p - t;
        loss += diff * diff;
        grad.data[i] = 2.0 * diff;
    }
    let cnt = pred.data.len() as f32;
    loss /= cnt;
    for g in grad.data.iter_mut() {
        *g /= cnt;
    }
    (loss, grad)
}

/// Kullback-Leibler divergence between a normal with mean `mu` and log-variance
/// `logvar` and the unit Gaussian. Returns the average loss and gradients for
/// `mu` and `logvar`.
pub fn kl_divergence(mu: &Matrix, logvar: &Matrix) -> (f32, Matrix, Matrix) {
    assert_eq!(mu.rows, logvar.rows);
    assert_eq!(mu.cols, logvar.cols);
    let mut grad_mu = Matrix::zeros(mu.rows, mu.cols);
    let mut grad_lv = Matrix::zeros(mu.rows, mu.cols);
    let mut loss = 0.0f32;
    for i in 0..mu.data.len() {
        let m = mu.data[i];
        let lv = logvar.data[i];
        let exp_lv = lv.exp();
        loss += -0.5 * (1.0 + lv - m * m - exp_lv);
        grad_mu.data[i] = m;
        grad_lv.data[i] = 0.5 * (exp_lv - 1.0);
    }
    let cnt = mu.data.len() as f32;
    loss /= cnt;
    for g in grad_mu.data.iter_mut() {
        *g /= cnt;
    }
    for g in grad_lv.data.iter_mut() {
        *g /= cnt;
    }
    (loss, grad_mu, grad_lv)
}
