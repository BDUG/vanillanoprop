#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn zeros(r: usize, c: usize) -> Self {
        Matrix { rows: r, cols: c, data: vec![0.0; r * c] }
    }

    pub fn from_vec(r: usize, c: usize, v: Vec<f32>) -> Self {
        assert_eq!(v.len(), r * c);
        Matrix { rows: r, cols: c, data: v }
    }

    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        self.data[r * self.cols + c] = v;
    }

    pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
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
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut v = vec![0.0; self.data.len()];
        for i in 0..v.len() {
            v[i] = self.data[i] + other.data[i];
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
