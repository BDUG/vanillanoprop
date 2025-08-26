use crate::math::Matrix;
use crate::tensor::Tensor;
use super::layer::Layer;
use super::linear::LinearT;
use super::{sigmoid, tanh};

fn elem_mul(a: &Matrix, b: &Matrix) -> Matrix {
    let mut v = vec![0.0; a.data.len()];
    for i in 0..v.len() {
        v[i] = a.data[i] * b.data[i];
    }
    Matrix::from_vec(a.rows, a.cols, v)
}

fn elem_sub(a: &Matrix, b: &Matrix) -> Matrix {
    let mut v = vec![0.0; a.data.len()];
    for i in 0..v.len() {
        v[i] = a.data[i] - b.data[i];
    }
    Matrix::from_vec(a.rows, a.cols, v)
}

fn elem_sub_from_one(a: &Matrix) -> Matrix {
    let mut v = vec![0.0; a.data.len()];
    for i in 0..v.len() {
        v[i] = 1.0 - a.data[i];
    }
    Matrix::from_vec(a.rows, a.cols, v)
}

fn concat_rows(rows: Vec<Matrix>) -> Matrix {
    if rows.is_empty() {
        return Matrix::zeros(0, 0);
    }
    let cols = rows[0].cols;
    let mut data = Vec::new();
    for m in rows {
        data.extend(m.data);
    }
    Matrix::from_vec(data.len() / cols, cols, data)
}

pub struct LSTM {
    pub w_ii: LinearT,
    pub w_if: LinearT,
    pub w_io: LinearT,
    pub w_ig: LinearT,
    pub w_hi: LinearT,
    pub w_hf: LinearT,
    pub w_ho: LinearT,
    pub w_hg: LinearT,
    cache: Vec<LSTMCache>,
    input_dim: usize,
    hidden_dim: usize,
}

struct LSTMCache {
    c: Matrix,
    i: Matrix,
    f: Matrix,
    o: Matrix,
    g: Matrix,
}

impl LSTM {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            w_ii: LinearT::new(input_dim, hidden_dim),
            w_if: LinearT::new(input_dim, hidden_dim),
            w_io: LinearT::new(input_dim, hidden_dim),
            w_ig: LinearT::new(input_dim, hidden_dim),
            w_hi: LinearT::new(hidden_dim, hidden_dim),
            w_hf: LinearT::new(hidden_dim, hidden_dim),
            w_ho: LinearT::new(hidden_dim, hidden_dim),
            w_hg: LinearT::new(hidden_dim, hidden_dim),
            cache: Vec::new(),
            input_dim,
            hidden_dim,
        }
    }

    fn step(&self, x_t: &Matrix, h_prev: &Matrix, c_prev: &Matrix) -> (Matrix, Matrix, Matrix, Matrix, Matrix, Matrix) {
        let mut i = Matrix::matmul(x_t, &self.w_ii.w.data)
            .add(&Matrix::matmul(h_prev, &self.w_hi.w.data));
        sigmoid::forward_matrix(&mut i);
        let mut f = Matrix::matmul(x_t, &self.w_if.w.data)
            .add(&Matrix::matmul(h_prev, &self.w_hf.w.data));
        sigmoid::forward_matrix(&mut f);
        let mut o = Matrix::matmul(x_t, &self.w_io.w.data)
            .add(&Matrix::matmul(h_prev, &self.w_ho.w.data));
        sigmoid::forward_matrix(&mut o);
        let mut g = Matrix::matmul(x_t, &self.w_ig.w.data)
            .add(&Matrix::matmul(h_prev, &self.w_hg.w.data));
        tanh::forward_matrix(&mut g);
        let c = elem_mul(&f, c_prev).add(&elem_mul(&i, &g));
        let mut h = c.clone();
        tanh::forward_matrix(&mut h);
        let h = elem_mul(&o, &h);
        (h, c, i, f, o, g)
    }

    fn step_train(&mut self, x_t: &Matrix, h_prev: &Matrix, c_prev: &Matrix) -> (Matrix, Matrix, Matrix, Matrix, Matrix, Matrix) {
        let mut i = self.w_ii.forward_local(x_t)
            .add(&self.w_hi.forward_local(h_prev));
        sigmoid::forward_matrix(&mut i);
        let mut f = self.w_if.forward_local(x_t)
            .add(&self.w_hf.forward_local(h_prev));
        sigmoid::forward_matrix(&mut f);
        let mut o = self.w_io.forward_local(x_t)
            .add(&self.w_ho.forward_local(h_prev));
        sigmoid::forward_matrix(&mut o);
        let mut g = self.w_ig.forward_local(x_t)
            .add(&self.w_hg.forward_local(h_prev));
        tanh::forward_matrix(&mut g);
        let c = elem_mul(&f, c_prev).add(&elem_mul(&i, &g));
        let mut h = c.clone();
        tanh::forward_matrix(&mut h);
        let h = elem_mul(&o, &h);
        (h, c, i, f, o, g)
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut h_prev = Matrix::zeros(1, self.hidden_dim);
        let mut c_prev = Matrix::zeros(1, self.hidden_dim);
        let mut outs = Vec::new();
        for t in 0..x.data.rows {
            let x_t = Matrix::from_vec(1, self.input_dim, x.data.data[t*self.input_dim..(t+1)*self.input_dim].to_vec());
            let (h_t, c_t, _, _, _, _) = self.step(&x_t, &h_prev, &c_prev);
            h_prev = h_t.clone();
            c_prev = c_t;
            outs.push(h_prev.clone());
        }
        Tensor::from_matrix(concat_rows(outs))
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        self.cache.clear();
        let mut h_prev = Matrix::zeros(1, self.hidden_dim);
        let mut c_prev = Matrix::zeros(1, self.hidden_dim);
        let mut outs = Vec::new();
        for t in 0..x.rows {
            let x_t = Matrix::from_vec(1, self.input_dim, x.data[t*self.input_dim..(t+1)*self.input_dim].to_vec());
            let (h_t, c_t, i, f, o, g) = self.step_train(&x_t, &h_prev, &c_prev);
            self.cache.push(LSTMCache { c: c_t.clone(), i, f, o, g });
            outs.push(h_t.clone());
            h_prev = h_t;
            c_prev = c_t;
        }
        concat_rows(outs)
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let seq_len = grad_out.rows;
        let mut grad_x = Matrix::zeros(seq_len, self.input_dim);
        let mut dh_next = Matrix::zeros(1, self.hidden_dim);
        let mut dc_next = Matrix::zeros(1, self.hidden_dim);
        let zero_c = Matrix::zeros(1, self.hidden_dim);
        for t in (0..seq_len).rev() {
            let cache = &self.cache[t];
            let c_prev = if t > 0 { &self.cache[t-1].c } else { &zero_c };
            let mut dh = Matrix::from_vec(1, self.hidden_dim, grad_out.data[t*self.hidden_dim..(t+1)*self.hidden_dim].to_vec());
            dh = dh.add(&dh_next);
            let mut tanh_c = cache.c.clone();
            tanh::forward_matrix(&mut tanh_c);
            let mut do_gate = elem_mul(&dh, &tanh_c);
            sigmoid::backward(&mut do_gate, &cache.o);
            let mut dc = elem_mul(&dh, &cache.o);
            for (dcv, tc) in dc.data.iter_mut().zip(tanh_c.data.iter()) {
                *dcv *= 1.0 - tc*tc;
            }
            dc = dc.add(&dc_next);
            let mut di = elem_mul(&dc, &cache.g);
            sigmoid::backward(&mut di, &cache.i);
            let mut df = elem_mul(&dc, c_prev);
            sigmoid::backward(&mut df, &cache.f);
            let mut dg = elem_mul(&dc, &cache.i);
            tanh::backward(&mut dg, &cache.g);
            let grad_x_i = self.w_ii.backward(&di);
            let grad_x_f = self.w_if.backward(&df);
            let grad_x_o = self.w_io.backward(&do_gate);
            let grad_x_g = self.w_ig.backward(&dg);
            let mut gx = grad_x_i.add(&grad_x_f);
            gx = gx.add(&grad_x_o);
            gx = gx.add(&grad_x_g);
            for j in 0..self.input_dim {
                grad_x.set(t, j, gx.get(0, j));
            }
            let grad_h_i = self.w_hi.backward(&di);
            let grad_h_f = self.w_hf.backward(&df);
            let grad_h_o = self.w_ho.backward(&do_gate);
            let grad_h_g = self.w_hg.backward(&dg);
            dh_next = grad_h_i.add(&grad_h_f);
            dh_next = dh_next.add(&grad_h_o);
            dh_next = dh_next.add(&grad_h_g);
            dc_next = elem_mul(&dc, &cache.f);
        }
        grad_x
    }

    pub fn zero_grad(&mut self) {
        self.w_ii.zero_grad();
        self.w_if.zero_grad();
        self.w_io.zero_grad();
        self.w_ig.zero_grad();
        self.w_hi.zero_grad();
        self.w_hf.zero_grad();
        self.w_ho.zero_grad();
        self.w_hg.zero_grad();
    }

    pub fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        let grad_x = self.backward(grad_out);
        let mut params = self.parameters();
        for p in params.iter_mut() {
            p.sgd_step(lr, 0.0);
        }
        grad_x
    }

    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.w_ii.adam_step(lr, beta1, beta2, eps, weight_decay);
        self.w_if.adam_step(lr, beta1, beta2, eps, weight_decay);
        self.w_io.adam_step(lr, beta1, beta2, eps, weight_decay);
        self.w_ig.adam_step(lr, beta1, beta2, eps, weight_decay);
        self.w_hi.adam_step(lr, beta1, beta2, eps, weight_decay);
        self.w_hf.adam_step(lr, beta1, beta2, eps, weight_decay);
        self.w_ho.adam_step(lr, beta1, beta2, eps, weight_decay);
        self.w_hg.adam_step(lr, beta1, beta2, eps, weight_decay);
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let (w_ii, w_if, w_io, w_ig, w_hi, w_hf, w_ho, w_hg) = (
            &mut self.w_ii,
            &mut self.w_if,
            &mut self.w_io,
            &mut self.w_ig,
            &mut self.w_hi,
            &mut self.w_hf,
            &mut self.w_ho,
            &mut self.w_hg,
        );
        vec![w_ii, w_if, w_io, w_ig, w_hi, w_hf, w_ho, w_hg]
    }
}

impl Layer for LSTM {
    fn forward(&self, x: &Tensor) -> Tensor { LSTM::forward(self, x) }
    fn forward_train(&mut self, x: &Matrix) -> Matrix { LSTM::forward_train(self, x) }
    fn backward(&mut self, grad_out: &Matrix) -> Matrix { LSTM::backward(self, grad_out) }
    fn zero_grad(&mut self) { LSTM::zero_grad(self); }
    fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix { LSTM::fa_update(self, grad_out, lr) }
    fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        LSTM::adam_step(self, lr, beta1, beta2, eps, weight_decay);
    }
    fn parameters(&mut self) -> Vec<&mut LinearT> { LSTM::parameters(self) }
}

pub struct GRU {
    pub w_ir: LinearT,
    pub w_iz: LinearT,
    pub w_in: LinearT,
    pub w_hr: LinearT,
    pub w_hz: LinearT,
    pub w_hn: LinearT,
    cache: Vec<GRUCache>,
    input_dim: usize,
    hidden_dim: usize,
}

struct GRUCache {
    h: Matrix,
    r: Matrix,
    z: Matrix,
    n: Matrix,
}

impl GRU {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            w_ir: LinearT::new(input_dim, hidden_dim),
            w_iz: LinearT::new(input_dim, hidden_dim),
            w_in: LinearT::new(input_dim, hidden_dim),
            w_hr: LinearT::new(hidden_dim, hidden_dim),
            w_hz: LinearT::new(hidden_dim, hidden_dim),
            w_hn: LinearT::new(hidden_dim, hidden_dim),
            cache: Vec::new(),
            input_dim,
            hidden_dim,
        }
    }

    fn step(&self, x_t: &Matrix, h_prev: &Matrix) -> (Matrix, Matrix, Matrix, Matrix) {
        let mut r = Matrix::matmul(x_t, &self.w_ir.w.data)
            .add(&Matrix::matmul(h_prev, &self.w_hr.w.data));
        sigmoid::forward_matrix(&mut r);
        let mut z = Matrix::matmul(x_t, &self.w_iz.w.data)
            .add(&Matrix::matmul(h_prev, &self.w_hz.w.data));
        sigmoid::forward_matrix(&mut z);
        let rh = elem_mul(&r, h_prev);
        let mut n = Matrix::matmul(x_t, &self.w_in.w.data)
            .add(&Matrix::matmul(&rh, &self.w_hn.w.data));
        tanh::forward_matrix(&mut n);
        let one_minus_z = elem_sub_from_one(&z);
        let h = elem_mul(&z, h_prev).add(&elem_mul(&one_minus_z, &n));
        (h, r, z, n)
    }

    fn step_train(&mut self, x_t: &Matrix, h_prev: &Matrix) -> (Matrix, Matrix, Matrix, Matrix) {
        let mut r = self.w_ir.forward_local(x_t)
            .add(&self.w_hr.forward_local(h_prev));
        sigmoid::forward_matrix(&mut r);
        let mut z = self.w_iz.forward_local(x_t)
            .add(&self.w_hz.forward_local(h_prev));
        sigmoid::forward_matrix(&mut z);
        let rh = elem_mul(&r, h_prev);
        let mut n = self.w_in.forward_local(x_t)
            .add(&self.w_hn.forward_local(&rh));
        tanh::forward_matrix(&mut n);
        let one_minus_z = elem_sub_from_one(&z);
        let h = elem_mul(&z, h_prev).add(&elem_mul(&one_minus_z, &n));
        (h, r, z, n)
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut h_prev = Matrix::zeros(1, self.hidden_dim);
        let mut outs = Vec::new();
        for t in 0..x.data.rows {
            let x_t = Matrix::from_vec(1, self.input_dim, x.data.data[t*self.input_dim..(t+1)*self.input_dim].to_vec());
            let (h_t, _, _, _) = self.step(&x_t, &h_prev);
            h_prev = h_t.clone();
            outs.push(h_prev.clone());
        }
        Tensor::from_matrix(concat_rows(outs))
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        self.cache.clear();
        let mut h_prev = Matrix::zeros(1, self.hidden_dim);
        let mut outs = Vec::new();
        for t in 0..x.rows {
            let x_t = Matrix::from_vec(1, self.input_dim, x.data[t*self.input_dim..(t+1)*self.input_dim].to_vec());
            let (h_t, r, z, n) = self.step_train(&x_t, &h_prev);
            self.cache.push(GRUCache { h: h_t.clone(), r, z, n });
            outs.push(h_t.clone());
            h_prev = h_t;
        }
        concat_rows(outs)
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let seq_len = grad_out.rows;
        let mut grad_x = Matrix::zeros(seq_len, self.input_dim);
        let mut dh_next = Matrix::zeros(1, self.hidden_dim);
        let zero_h = Matrix::zeros(1, self.hidden_dim);
        for t in (0..seq_len).rev() {
            let cache = &self.cache[t];
            let h_prev = if t > 0 { &self.cache[t-1].h } else { &zero_h };
            let mut dh = Matrix::from_vec(1, self.hidden_dim, grad_out.data[t*self.hidden_dim..(t+1)*self.hidden_dim].to_vec());
            dh = dh.add(&dh_next);
            let mut dn = elem_mul(&dh, &elem_sub_from_one(&cache.z));
            tanh::backward(&mut dn, &cache.n);
            let drh = self.w_hn.backward(&dn);
            let grad_x_n = self.w_in.backward(&dn);
            let mut dh_prev = elem_mul(&drh, &cache.r);
            let mut dr = elem_mul(&drh, h_prev);
            sigmoid::backward(&mut dr, &cache.r);
            let grad_x_r = self.w_ir.backward(&dr);
            let grad_h_r = self.w_hr.backward(&dr);
            dh_prev = dh_prev.add(&grad_h_r);
            let mut dz = elem_mul(&dh, &elem_sub(h_prev, &cache.n));
            sigmoid::backward(&mut dz, &cache.z);
            let grad_x_z = self.w_iz.backward(&dz);
            let grad_h_z = self.w_hz.backward(&dz);
            dh_prev = dh_prev.add(&grad_h_z);
            dh_prev = dh_prev.add(&elem_mul(&dh, &cache.z));
            let mut gx = grad_x_n.add(&grad_x_r);
            gx = gx.add(&grad_x_z);
            for j in 0..self.input_dim {
                grad_x.set(t, j, gx.get(0, j));
            }
            dh_next = dh_prev;
        }
        grad_x
    }

    pub fn zero_grad(&mut self) {
        self.w_ir.zero_grad();
        self.w_iz.zero_grad();
        self.w_in.zero_grad();
        self.w_hr.zero_grad();
        self.w_hz.zero_grad();
        self.w_hn.zero_grad();
    }

    pub fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix {
        let grad_x = self.backward(grad_out);
        let mut params = self.parameters();
        for p in params.iter_mut() {
            p.sgd_step(lr, 0.0);
        }
        grad_x
    }

    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.w_ir.adam_step(lr, beta1, beta2, eps, weight_decay);
        self.w_iz.adam_step(lr, beta1, beta2, eps, weight_decay);
        self.w_in.adam_step(lr, beta1, beta2, eps, weight_decay);
        self.w_hr.adam_step(lr, beta1, beta2, eps, weight_decay);
        self.w_hz.adam_step(lr, beta1, beta2, eps, weight_decay);
        self.w_hn.adam_step(lr, beta1, beta2, eps, weight_decay);
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let (w_ir, w_iz, w_in, w_hr, w_hz, w_hn) = (
            &mut self.w_ir,
            &mut self.w_iz,
            &mut self.w_in,
            &mut self.w_hr,
            &mut self.w_hz,
            &mut self.w_hn,
        );
        vec![w_ir, w_iz, w_in, w_hr, w_hz, w_hn]
    }
}

impl Layer for GRU {
    fn forward(&self, x: &Tensor) -> Tensor { GRU::forward(self, x) }
    fn forward_train(&mut self, x: &Matrix) -> Matrix { GRU::forward_train(self, x) }
    fn backward(&mut self, grad_out: &Matrix) -> Matrix { GRU::backward(self, grad_out) }
    fn zero_grad(&mut self) { GRU::zero_grad(self); }
    fn fa_update(&mut self, grad_out: &Matrix, lr: f32) -> Matrix { GRU::fa_update(self, grad_out, lr) }
    fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        GRU::adam_step(self, lr, beta1, beta2, eps, weight_decay);
    }
    fn parameters(&mut self) -> Vec<&mut LinearT> { GRU::parameters(self) }
}

