use vanillanoprop::math::Matrix;

fn softmax_reference(m: &Matrix) -> Matrix {
    let mut v = vec![0.0; m.data.len()];
    for (out_row, row_slice) in v.chunks_mut(m.cols).zip(m.data.chunks(m.cols)) {
        let max = row_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row_slice.iter().map(|x| (*x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (out, e) in out_row.iter_mut().zip(exps.iter()) {
            *out = e / sum;
        }
    }
    Matrix::from_vec(m.rows, m.cols, v)
}

#[test]
fn softmax_matches_reference() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0]);
    let expected = softmax_reference(&m);
    let actual = m.softmax();
    for (a, b) in actual.data.iter().zip(expected.data.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn softmax_matches_reference_parallel() {
    // Larger matrix to trigger the parallel SIMD path.
    let rows = 64;
    let cols = 10;
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| (i % cols) as f32 - 5.0)
        .collect();
    let m = Matrix::from_vec(rows, cols, data);
    let expected = softmax_reference(&m);
    let actual = m.softmax();
    for (a, b) in actual.data.iter().zip(expected.data.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn softmax_rows_sum_to_one() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0]);
    let sm = m.softmax();
    for row in sm.data.chunks(sm.cols) {
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
