use crate::device::{Cpu, Device};
use crate::tensor::Tensor;
use nalgebra::DMatrix;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "cuda")]
use cust::prelude::*;
#[cfg(feature = "cuda")]
use nvrtc::NvrtcProgram;

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

/// A lightweight view into a matrix's data that avoids allocation by
/// borrowing from the original matrix.  The view supports arbitrary row and
/// column strides which allows representing transposed matrices and slices of
/// rows or columns without copying.
#[derive(Clone, Copy, Debug)]
pub struct MatrixView<'a> {
    pub rows: usize,
    pub cols: usize,
    data: &'a [f32],
    row_stride: usize,
    col_stride: usize,
    offset: usize,
}

impl<'a> MatrixView<'a> {
    /// Return the element at the specified row and column.
    #[inline(always)]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        let idx = self.offset + r * self.row_stride + c * self.col_stride;
        self.data[idx]
    }

    /// Create a transposed view of this matrix without copying data.
    pub fn transpose(&self) -> MatrixView<'a> {
        MatrixView {
            rows: self.cols,
            cols: self.rows,
            data: self.data,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
            offset: self.offset,
        }
    }
}

/// Trait abstracting over owned matrices and borrowed matrix views.
pub trait MatrixLike {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn get(&self, r: usize, c: usize) -> f32;
}

impl MatrixLike for Matrix {
    #[inline(always)]
    fn rows(&self) -> usize {
        self.rows
    }
    #[inline(always)]
    fn cols(&self) -> usize {
        self.cols
    }
    #[inline(always)]
    fn get(&self, r: usize, c: usize) -> f32 {
        self.get(r, c)
    }
}

impl<'a> MatrixLike for MatrixView<'a> {
    #[inline(always)]
    fn rows(&self) -> usize {
        self.rows
    }
    #[inline(always)]
    fn cols(&self) -> usize {
        self.cols
    }
    #[inline(always)]
    fn get(&self, r: usize, c: usize) -> f32 {
        self.get(r, c)
    }
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

    let m = a.rows;
    let n = b.cols;
    let k_dim = a.cols;
    let mut out = vec![0.0; m * n];

    #[cfg(feature = "matrixmultiply")]
    unsafe {
        // Row-major matrix multiplication using the `matrixmultiply` backend
        matrixmultiply::sgemm(
            m,
            k_dim,
            n,
            1.0,
            a.data.as_ptr(),
            k_dim as isize,
            1,
            b.data.as_ptr(),
            n as isize,
            1,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }

    #[cfg(not(feature = "matrixmultiply"))]
    {
        const BLOCK: usize = 32;
        const PAR_THRESHOLD: usize = 128 * 128; // Use rayon when matrix is reasonably large

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
    }

    Matrix::from_vec(m, n, out)
}

#[cfg(feature = "cuda")]
pub(crate) fn matmul_cuda(a: &Matrix, b: &Matrix) -> Matrix {
    let muls = a.rows * a.cols * b.cols;
    let adds = muls;
    inc_mul_ops_by(muls);
    inc_add_ops_by(adds);
    assert_eq!(a.cols, b.rows);

    let m = a.rows as i32;
    let n = b.cols as i32;
    let k_dim = a.cols as i32;

    let src = r#"
extern "C" __global__ void matmul(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}
"#;

    let prog = NvrtcProgram::new(src, None, &[], &[]).expect("nvrtc program");
    prog.compile(&[]).expect("nvrtc compile");
    let ptx = prog.get_ptx().expect("nvrtc get ptx");

    let _ctx = cust::quick_init().expect("CUDA init failed");
    let module = Module::from_ptx(ptx.as_str(), &[]).expect("module from PTX");
    let stream = Stream::new(StreamFlags::DEFAULT, None).expect("stream");
    let func = module.get_function("matmul").expect("function");

    let mut d_a = DeviceBuffer::from_slice(&a.data).expect("device buffer A");
    let mut d_b = DeviceBuffer::from_slice(&b.data).expect("device buffer B");
    let mut d_c = DeviceBuffer::<f32>::zeroed((m * n) as usize).expect("device buffer C");

    let block = (16u32, 16u32, 1u32);
    let grid = (
        ((n as u32 + block.0 - 1) / block.0),
        ((m as u32 + block.1 - 1) / block.1),
        1u32,
    );

    unsafe {
        launch!(func<<<grid, block, 0, stream>>>(
            d_a.as_device_ptr(),
            d_b.as_device_ptr(),
            d_c.as_device_ptr(),
            m,
            n,
            k_dim
        ))
        .expect("kernel launch failed");
    }
    stream.synchronize().expect("stream sync");

    let mut out = vec![0.0f32; (m * n) as usize];
    d_c.copy_to(&mut out).expect("copy from device");

    Matrix::from_vec(m as usize, n as usize, out)
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

    /// Borrow a range of columns without allocating.
    pub fn view_cols(&self, start: usize, width: usize) -> MatrixView {
        assert!(start + width <= self.cols);
        MatrixView {
            rows: self.rows,
            cols: width,
            data: &self.data,
            row_stride: self.cols,
            col_stride: 1,
            offset: start,
        }
    }

    /// Borrow a range of rows without allocating.
    pub fn view_rows(&self, start: usize, count: usize) -> MatrixView {
        assert!(start + count <= self.rows);
        MatrixView {
            rows: count,
            cols: self.cols,
            data: &self.data,
            row_stride: self.cols,
            col_stride: 1,
            offset: start * self.cols,
        }
    }

    /// Multiply `a` and `b` using the default [`Cpu`] device.
    pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
        Cpu.matmul(a, b)
    }

    /// Multiply matrices or views without requiring ownership.
    pub fn matmul_views<A: MatrixLike, B: MatrixLike>(a: &A, b: &B) -> Matrix {
        assert_eq!(a.cols(), b.rows());
        let m = a.rows();
        let n = b.cols();
        let k_dim = a.cols();
        let muls = m * k_dim * n;
        let adds = muls;
        inc_mul_ops_by(muls);
        inc_add_ops_by(adds);
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for k in 0..k_dim {
                let a_val = a.get(i, k);
                for j in 0..n {
                    out[i * n + j] += a_val * b.get(k, j);
                }
            }
        }
        Matrix::from_vec(m, n, out)
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

    /// Compute the singular value decomposition of the matrix using the
    /// [`nalgebra`](https://crates.io/crates/nalgebra) crate.
    ///
    /// Returns matrices `U`, `S`, and `Vt` such that `self = U * S * Vt`.
    /// The `S` matrix is square with dimension `min(rows, cols)` and contains
    /// the singular values on its diagonal.
    ///
    /// # Examples
    /// ```
    /// use vanillanoprop::math::Matrix;
    ///
    /// let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    /// let (u, s, vt) = m.svd();
    /// let us = Matrix::matmul(&u, &s);
    /// let reconstructed = Matrix::matmul(&us, &vt);
    /// for (a, b) in reconstructed.data.iter().zip(m.data.iter()) {
    ///     assert!((a - b).abs() < 1e-4);
    /// }
    /// ```
    pub fn svd(&self) -> (Matrix, Matrix, Matrix) {
        let m = DMatrix::<f32>::from_row_slice(self.rows, self.cols, &self.data);
        let svd = m.svd(true, true);
        let u = svd.u.expect("SVD failed: missing U");
        let vt = svd.v_t.expect("SVD failed: missing Vt");
        let s = svd.singular_values;

        let mut u_vec = Vec::with_capacity(u.nrows() * u.ncols());
        for r in 0..u.nrows() {
            for c in 0..u.ncols() {
                u_vec.push(u[(r, c)]);
            }
        }
        let u_mat = Matrix::from_vec(u.nrows(), u.ncols(), u_vec);

        let mut vt_vec = Vec::with_capacity(vt.nrows() * vt.ncols());
        for r in 0..vt.nrows() {
            for c in 0..vt.ncols() {
                vt_vec.push(vt[(r, c)]);
            }
        }
        let vt_mat = Matrix::from_vec(vt.nrows(), vt.ncols(), vt_vec);

        let rank = s.len();
        let mut s_mat = Matrix::zeros(rank, rank);
        for i in 0..rank {
            s_mat.set(i, i, s[i]);
        }

        (u_mat, s_mat, vt_mat)
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

    // Number of rows we can actually process
    let rows_to_process = targets.len().min(logits.rows.saturating_sub(row_offset));
    let cols = logits.cols;

    // Slices covering only the rows that will be processed
    let logits_slice = &logits.data[row_offset * cols..row_offset * cols + rows_to_process * cols];
    let grad_slice = &mut grad.data[row_offset * cols..row_offset * cols + rows_to_process * cols];

    // Process each row in parallel, computing per-row loss and prediction.
    let results: Vec<(f32, usize)> = grad_slice
        .par_chunks_mut(cols)
        .zip(logits_slice.par_chunks(cols))
        .enumerate()
        .map(|(i, (grad_row, row_slice))| {
            let tok = targets[i];

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

            // Second pass: compute exponentials and their sum while storing
            // them directly into the gradient buffer.
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

            (-(target_prob + 1e-9).ln(), best_tok)
        })
        .collect();

    let mut loss: f32 = results.iter().map(|(l, _)| *l).sum();
    let preds: Vec<usize> = results.into_iter().map(|(_, p)| p).collect();
    let cnt = rows_to_process as f32;

    if cnt > 0.0 {
        loss /= cnt;
        grad.data.par_iter_mut().for_each(|v| *v /= cnt);
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

// ---------------------------------------------------------------------------
// Tensor based routines
// ---------------------------------------------------------------------------

/// Determine the broadcasted shape of two tensors following NumPy rules.
fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let rank = a.len().max(b.len());
    let mut shape = Vec::with_capacity(rank);
    for i in 0..rank {
        let ad = *a
            .get(a.len().wrapping_sub(1).saturating_sub(i))
            .unwrap_or(&1);
        let bd = *b
            .get(b.len().wrapping_sub(1).saturating_sub(i))
            .unwrap_or(&1);
        assert!(ad == bd || ad == 1 || bd == 1, "incompatible shapes");
        shape.push(ad.max(bd));
    }
    shape.reverse();
    shape
}

/// Elementwise addition supporting broadcasting.
pub fn tensor_add(a: &Tensor, b: &Tensor) -> Tensor {
    let shape = broadcast_shape(&a.shape, &b.shape);
    let rank = shape.len();
    let out_len: usize = shape.iter().product();
    let mut out = vec![0.0; out_len];

    // Prepare padded shapes so both tensors share the same rank.
    let mut a_shape = vec![1; rank];
    a_shape[rank - a.shape.len()..].copy_from_slice(&a.shape);
    let mut b_shape = vec![1; rank];
    b_shape[rank - b.shape.len()..].copy_from_slice(&b.shape);

    // Compute strides with zero stride for broadcast dimensions.
    let mut a_stride = vec![0; rank];
    let mut s = 1;
    for i in (0..rank).rev() {
        if a_shape[i] != 1 {
            a_stride[i] = s;
        }
        s *= a_shape[i];
    }
    let mut b_stride = vec![0; rank];
    s = 1;
    for i in (0..rank).rev() {
        if b_shape[i] != 1 {
            b_stride[i] = s;
        }
        s *= b_shape[i];
    }

    // Iterate over the output tensor computing indices on the fly.
    let mut a_idx = 0usize;
    let mut b_idx = 0usize;
    let mut counters = vec![0usize; rank];
    for i in 0..out_len {
        out[i] = a.data[a_idx] + b.data[b_idx];

        // Update indices in a row-major fashion.
        for d in (0..rank).rev() {
            counters[d] += 1;
            if counters[d] < shape[d] {
                a_idx += a_stride[d];
                b_idx += b_stride[d];
                break;
            }
            counters[d] = 0;
            a_idx -= a_stride[d] * (shape[d] - 1);
            b_idx -= b_stride[d] * (shape[d] - 1);
        }
    }

    Tensor { data: out, shape }
}

/// Multiply two rank-2 tensors.
///
/// When the `matrixmultiply` feature (or another BLAS backend) is enabled the
/// computation is delegated to [`matmul_cpu`], ensuring we benefit from any
/// optimized routines. Otherwise a simple triple loop implementation is used
/// with optional `rayon` parallelisation when the matrices are large enough.
pub fn tensor_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape.len(), 2, "lhs must be rank 2");
    assert_eq!(b.shape.len(), 2, "rhs must be rank 2");
    let m = a.shape[0];
    let k = a.shape[1];
    let k2 = b.shape[0];
    let n = b.shape[1];
    assert_eq!(k, k2, "matmul dimension mismatch");

    #[cfg(feature = "matrixmultiply")]
    {
        let a_m = Matrix::from_vec(m, k, a.data.clone());
        let b_m = Matrix::from_vec(k, n, b.data.clone());
        return Tensor::from_matrix(matmul_cpu(&a_m, &b_m));
    }

    #[cfg(not(feature = "matrixmultiply"))]
    {
        const PAR_THRESHOLD: usize = 128 * 128;
        let mut out = vec![0.0; m * n];

        if m * n > PAR_THRESHOLD {
            out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                for j in 0..n {
                    let mut sum = 0.0;
                    for p in 0..k {
                        sum += a.data[i * k + p] * b.data[p * n + j];
                    }
                    row[j] = sum;
                }
            });
        } else {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for p in 0..k {
                        sum += a.data[i * k + p] * b.data[p * n + j];
                    }
                    out[i * n + j] = sum;
                }
            }
        }

        Tensor {
            data: out,
            shape: vec![m, n],
        }
    }
}

/// Transpose a rank-2 tensor.
pub fn tensor_transpose(t: &Tensor) -> Tensor {
    assert_eq!(t.shape.len(), 2);
    let rows = t.shape[0];
    let cols = t.shape[1];
    let mut out = vec![0.0; t.data.len()];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = t.data[i * cols + j];
        }
    }
    Tensor {
        data: out,
        shape: vec![cols, rows],
    }
}

/// Softmax along the last dimension of the tensor operating in-place on `t`.
pub fn tensor_softmax_inplace(t: &mut Tensor) {
    let cols = *t.shape.last().unwrap();
    let rows = t.data.len() / cols;
    for r in 0..rows {
        let start = r * cols;
        let row = &mut t.data[start..start + cols];
        let mut max = f32::NEG_INFINITY;
        for &v in row.iter() {
            max = f32::max(max, v);
        }
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

/// Softmax along the last dimension writing the result into `out` while
/// leaving `t` unchanged.
pub fn tensor_softmax_into(out: &mut Tensor, t: &Tensor) {
    assert_eq!(out.shape, t.shape);
    out.data.copy_from_slice(&t.data);
    tensor_softmax_inplace(out);
}

/// Allocate a new tensor containing the softmax result.
pub fn tensor_softmax(t: &Tensor) -> Tensor {
    let mut out = t.clone();
    tensor_softmax_inplace(&mut out);
    out
}

/// Compute softmax cross entropy loss and gradient w.r.t. logits.
pub fn tensor_softmax_cross_entropy(
    logits: &Tensor,
    targets: &[usize],
    row_offset: usize,
) -> (f32, Tensor) {
    assert_eq!(logits.shape.len(), 2);
    let rows = logits.shape[0];
    let cols = logits.shape[1];
    let mut grad = logits.clone();
    tensor_softmax_inplace(&mut grad);
    let mut loss = 0.0f32;
    for r in 0..rows {
        let target = targets[row_offset + r];
        let idx = r * cols + target;
        loss += -grad.data[idx].ln();
        grad.data[idx] -= 1.0;
    }
    loss /= rows as f32;
    for g in grad.data.iter_mut() {
        *g /= rows as f32;
    }
    (loss, grad)
}
