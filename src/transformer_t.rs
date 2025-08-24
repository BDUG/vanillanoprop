use crate::autograd::Tensor;
use crate::math::Matrix;
use crate::positional::positional_encoding;

// Simple linear module with rudimentary autograd support.  During training
// each `LinearT` stores the last input that was seen so that a backward pass
// can compute gradients for both the input and the weight matrix.  In
// addition, the struct keeps Adam optimizer statistics so that the optimizer
// state automatically persists across iterations.

pub struct LinearT {
    pub w: Tensor,
    grad: Matrix,
    m: Matrix,
    v: Matrix,
    t: usize,
    last_x: Matrix,
    fb: Matrix,
}

impl LinearT {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let data = Matrix::from_vec(
            in_dim,
            out_dim,
            (0..in_dim * out_dim)
                .map(|_| (rand::random::<f32>() - 0.5) * 0.02)
                .collect(),
        );
        let w = Tensor::from_matrix(data);
        let grad = Matrix::zeros(w.data.rows, w.data.cols);
        let m = Matrix::zeros(w.data.rows, w.data.cols);
        let v = Matrix::zeros(w.data.rows, w.data.cols);
        let last_x = Matrix::zeros(0, 0);
        let fb = Matrix::zeros(0, 0);
        Self {
            w,
            grad,
            m,
            v,
            t: 0,
            last_x,
            fb,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::matmul(x, &self.w)
    }

    /// Training variant of forward that remembers the input for backprop.
    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        self.last_x = x.clone();
        Matrix::matmul(x, &self.w.data)
    }

    /// Forward pass for local learning.
    pub fn forward_local(&mut self, x: &Matrix) -> Matrix {
        self.last_x = x.clone();
        Matrix::matmul(x, &self.w.data)
    }

    /// Backward pass for the linear layer.  Takes the gradient of the output
    /// and returns the gradient with respect to the input while storing the
    /// gradient for the weights inside `self.grad`.
    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let x_t = self.last_x.transpose();
        self.grad = Matrix::matmul(&x_t, grad_out);
        Matrix::matmul(grad_out, &self.w.data.transpose())
    }

    pub fn zero_grad(&mut self) {
        self.grad = Matrix::zeros(self.grad.rows, self.grad.cols);
    }

    /// Adam optimisation step.  This is intentionally very small and only
    /// implements what is required for the training examples in this
    /// repository.
    pub fn sgd_step(&mut self, lr: f32, weight_decay: f32) {
        for i in 0..self.grad.data.len() {
            let g = self.grad.data[i] + weight_decay * self.w.data.data[i];
            self.w.data.data[i] -= lr * g;
        }
    }

    pub fn adam_step(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) {
        self.t += 1;
        for i in 0..self.grad.data.len() {
            let g = self.grad.data[i] + weight_decay * self.w.data.data[i];
            self.m.data[i] = beta1 * self.m.data[i] + (1.0 - beta1) * g;
            self.v.data[i] = beta2 * self.v.data[i] + (1.0 - beta2) * g * g;
            let m_hat = self.m.data[i] / (1.0 - beta1.powi(self.t as i32));
            let v_hat = self.v.data[i] / (1.0 - beta2.powi(self.t as i32));
            self.w.data.data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        vec![self]
    }

    /// Local Hebbian-style weight update using a random feedback matrix.
    pub fn local_update(&mut self, top_error: &Matrix, lr: f32, weight_decay: f32) {
        if self.fb.rows != top_error.cols || self.fb.cols != self.w.data.cols {
            self.fb = Matrix::from_vec(
                top_error.cols,
                self.w.data.cols,
                (0..top_error.cols * self.w.data.cols)
                    .map(|_| (rand::random::<f32>() - 0.5) * 0.1)
                    .collect(),
            );
        }
        let local_err = Matrix::matmul(top_error, &self.fb);
        let x_t = self.last_x.transpose();
        let grad = Matrix::matmul(&x_t, &local_err);
        for i in 0..self.w.data.data.len() {
            let g = grad.data[i] + weight_decay * self.w.data.data[i];
            self.w.data.data[i] -= lr * g;
        }
    }
}

/// Embedding layer: maps one-hot (vocab_size) into dense model_dim.
pub struct EmbeddingT {
    pub table: LinearT, // weight matrix (vocab_size x model_dim)
}

impl EmbeddingT {
    pub fn new(vocab_size: usize, model_dim: usize) -> Self {
        Self {
            table: LinearT::new(vocab_size, model_dim),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // inference helper
        Tensor::matmul(x, &self.table.w)
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        self.table.forward_train(x)
    }

    pub fn forward_local(&mut self, x: &Matrix) -> Matrix {
        self.table.forward_local(x)
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        self.table.backward(grad_out)
    }

    pub fn zero_grad(&mut self) {
        self.table.zero_grad();
    }

    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.table
            .adam_step(lr, beta1, beta2, eps, weight_decay);
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        self.table.parameters()
    }

    pub fn local_update(&mut self, top_error: &Matrix, lr: f32, weight_decay: f32) {
        self.table.local_update(top_error, lr, weight_decay);
    }
}

pub struct FeedForwardT {
    pub w1: LinearT,
    pub w2: LinearT,
    // caches for backward
    mask: Vec<f32>,
    h1: Matrix,
}

impl FeedForwardT {
    pub fn new(dim: usize, hidden: usize) -> Self {
        Self {
            w1: LinearT::new(dim, hidden),
            w2: LinearT::new(hidden, dim),
            mask: Vec::new(),
            h1: Matrix::zeros(0, 0),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = self.w1.forward(x);
        for v in h.data.data.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        self.w2.forward(&h)
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        let mut h1 = self.w1.forward_train(x);
        let mut mask = vec![0.0; h1.data.len()];
        for (i, v) in h1.data.iter_mut().enumerate() {
            if *v < 0.0 {
                *v = 0.0;
            } else {
                mask[i] = 1.0;
            }
        }
        // store mask in w1.last_x? we cannot; we need to store for backward.
        // We'll temporarily store in last_x of w2? Instead easier: keep mask as
        // member variable of FeedForwardT.
        self.mask = mask;
        let out = self.w2.forward_train(&h1);
        self.h1 = h1; // store activated h1 for backward
        out
    }

    pub fn forward_local(&mut self, x: &Matrix) -> Matrix {
        let mut h1 = self.w1.forward_local(x);
        for v in h1.data.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        self.h1 = h1.clone();
        self.w2.forward_local(&h1)
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let grad_h1 = self.w2.backward(grad_out);
        let mut g = grad_h1.clone();
        for (i, v) in g.data.iter_mut().enumerate() {
            *v *= self.mask[i];
        }
        self.w1.backward(&g)
    }

    pub fn zero_grad(&mut self) {
        self.w1.zero_grad();
        self.w2.zero_grad();
    }

    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.w1
            .adam_step(lr, beta1, beta2, eps, weight_decay);
        self.w2
            .adam_step(lr, beta1, beta2, eps, weight_decay);
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let (w1, w2) = (&mut self.w1, &mut self.w2);
        vec![w1, w2]
    }

    pub fn local_update(&mut self, top_error: &Matrix, lr: f32, weight_decay: f32) {
        self.w1.local_update(top_error, lr, weight_decay);
        self.w2.local_update(top_error, lr, weight_decay);
    }
}

pub struct MultiHeadAttentionT {
    pub wq: LinearT,
    pub wk: LinearT,
    pub wv: LinearT,
    pub wo: LinearT,
    // caches for backward
    x: Matrix,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    attn: Matrix,
    scores: Matrix,
}

impl MultiHeadAttentionT {
    pub fn new(model_dim: usize) -> Self {
        Self {
            wq: LinearT::new(model_dim, model_dim),
            wk: LinearT::new(model_dim, model_dim),
            wv: LinearT::new(model_dim, model_dim),
            wo: LinearT::new(model_dim, model_dim),
            x: Matrix::zeros(0, 0),
            q: Matrix::zeros(0, 0),
            k: Matrix::zeros(0, 0),
            v: Matrix::zeros(0, 0),
            attn: Matrix::zeros(0, 0),
            scores: Matrix::zeros(0, 0),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);

        // ⚠️ stark vereinfachte Attention: keine Aufteilung in Köpfe,
        // kein Softmax, kein Masking – nur MatMul zur Demo
        let kt = Tensor::transpose(&k);
        let attn = Tensor::matmul(&q, &kt);
        let scores = Tensor::matmul(&attn, &v);
        self.wo.forward(&scores)
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        self.x = x.clone();
        self.q = self.wq.forward_train(x);
        self.k = self.wk.forward_train(x);
        self.v = self.wv.forward_train(x);
        let kt = self.k.transpose();
        self.attn = Matrix::matmul(&self.q, &kt);
        self.scores = Matrix::matmul(&self.attn, &self.v);
        self.wo.forward_train(&self.scores)
    }

    pub fn forward_local(&mut self, x: &Matrix) -> Matrix {
        self.x = x.clone();
        self.q = self.wq.forward_local(x);
        self.k = self.wk.forward_local(x);
        self.v = self.wv.forward_local(x);
        let kt = self.k.transpose();
        self.attn = Matrix::matmul(&self.q, &kt);
        self.scores = Matrix::matmul(&self.attn, &self.v);
        self.wo.forward_local(&self.scores)
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let grad_scores = self.wo.backward(grad_out);
        let grad_attn = Matrix::matmul(&grad_scores, &self.v.transpose());
        let grad_v = Matrix::matmul(&self.attn.transpose(), &grad_scores);
        let grad_q = Matrix::matmul(&grad_attn, &self.k);
        let grad_k = Matrix::matmul(&grad_attn.transpose(), &self.q);
        let gx_q = self.wq.backward(&grad_q);
        let gx_k = self.wk.backward(&grad_k);
        let gx_v = self.wv.backward(&grad_v);
        let tmp = Matrix::add(&gx_q, &gx_k);
        Matrix::add(&tmp, &gx_v)
    }

    pub fn zero_grad(&mut self) {
        self.wq.zero_grad();
        self.wk.zero_grad();
        self.wv.zero_grad();
        self.wo.zero_grad();
    }

    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.wq
            .adam_step(lr, beta1, beta2, eps, weight_decay);
        self.wk
            .adam_step(lr, beta1, beta2, eps, weight_decay);
        self.wv
            .adam_step(lr, beta1, beta2, eps, weight_decay);
        self.wo
            .adam_step(lr, beta1, beta2, eps, weight_decay);
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let (wq, wk, wv, wo) = (&mut self.wq, &mut self.wk, &mut self.wv, &mut self.wo);
        vec![wq, wk, wv, wo]
    }

    pub fn local_update(&mut self, top_error: &Matrix, lr: f32, weight_decay: f32) {
        self.wq.local_update(top_error, lr, weight_decay);
        self.wk.local_update(top_error, lr, weight_decay);
        self.wv.local_update(top_error, lr, weight_decay);
        self.wo.local_update(top_error, lr, weight_decay);
    }
}

pub struct EncoderLayerT {
    pub attn: MultiHeadAttentionT,
    pub ff: FeedForwardT,
    attn_out: Matrix,
}

impl EncoderLayerT {
    pub fn new(dim: usize, hidden: usize) -> Self {
        Self {
            attn: MultiHeadAttentionT::new(dim),
            ff: FeedForwardT::new(dim, hidden),
            attn_out: Matrix::zeros(0, 0),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.attn.forward(x);
        self.ff.forward(&h)
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        self.attn_out = self.attn.forward_train(x);
        self.ff.forward_train(&self.attn_out)
    }

    pub fn forward_local(&mut self, x: &Matrix) -> Matrix {
        self.attn_out = self.attn.forward_local(x);
        self.ff.forward_local(&self.attn_out)
    }

    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let grad_attn_out = self.ff.backward(grad_out);
        self.attn.backward(&grad_attn_out)
    }

    pub fn zero_grad(&mut self) {
        self.attn.zero_grad();
        self.ff.zero_grad();
    }

    pub fn adam_step(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) {
        self.attn
            .adam_step(lr, beta1, beta2, eps, weight_decay);
        self.ff
            .adam_step(lr, beta1, beta2, eps, weight_decay);
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let mut params = self.attn.parameters();
        params.extend(self.ff.parameters());
        params
    }

    pub fn local_update(&mut self, top_error: &Matrix, lr: f32, weight_decay: f32) {
        self.attn.local_update(top_error, lr, weight_decay);
        self.ff.local_update(top_error, lr, weight_decay);
    }
}

pub struct EncoderT {
    pub layers: Vec<EncoderLayerT>,
    pub embedding: EmbeddingT,
    pos: Matrix,
}

impl EncoderT {
    pub fn new(n: usize, vocab_size: usize, model_dim: usize, hidden: usize) -> Self {
        let mut v = Vec::new();
        for _ in 0..n {
            v.push(EncoderLayerT::new(model_dim, hidden));
        }
        Self {
            layers: v,
            embedding: EmbeddingT::new(vocab_size, model_dim),
            pos: Matrix::zeros(0, 0),
        }
    }

    pub fn forward(&self, x: &Matrix) -> Tensor {
        // existing inference path
        let mut h = Tensor::from_matrix(x.clone());
        h = self.embedding.forward(&h);
        let pos = positional_encoding(h.data.rows, h.data.cols);
        let p = Tensor::from_matrix(pos);
        h = Tensor::add(&h, &p);
        for l in &self.layers {
            h = l.forward(&h);
        }
        h
    }

    pub fn forward_train(&mut self, x: &Matrix) -> Matrix {
        let mut h = self.embedding.forward_train(x);
        self.pos = positional_encoding(h.rows, h.cols);
        h = Matrix::add(&h, &self.pos);
        for l in self.layers.iter_mut() {
            h = l.forward_train(&h);
        }
        h
    }

    pub fn forward_local(&mut self, x: &Matrix) -> Matrix {
        let mut h = self.embedding.forward_local(x);
        self.pos = positional_encoding(h.rows, h.cols);
        h = Matrix::add(&h, &self.pos);
        for l in self.layers.iter_mut() {
            h = l.forward_local(&h);
        }
        h
    }

    pub fn backward(&mut self, grad_out: &Matrix) {
        let mut g = grad_out.clone();
        for l in self.layers.iter_mut().rev() {
            g = l.backward(&g);
        }
        self.embedding.backward(&g);
    }

    pub fn zero_grad(&mut self) {
        self.embedding.zero_grad();
        for l in self.layers.iter_mut() {
            l.zero_grad();
        }
    }

    pub fn adam_step(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) {
        self.embedding
            .adam_step(lr, beta1, beta2, eps, weight_decay);
        for l in self.layers.iter_mut() {
            l.adam_step(lr, beta1, beta2, eps, weight_decay);
        }
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let mut params = self.embedding.parameters();
        for l in self.layers.iter_mut() {
            params.extend(l.parameters());
        }
        params
    }

    pub fn local_update(&mut self, top_error: &Matrix, lr: f32, weight_decay: f32) {
        self.embedding.local_update(top_error, lr, weight_decay);
        for l in self.layers.iter_mut() {
            l.local_update(top_error, lr, weight_decay);
        }
    }
}

pub struct DecoderLayerT {
    self_attn: MultiHeadAttentionT,
    enc_dec_attn: MultiHeadAttentionT,
    ff: FeedForwardT,
    h1: Matrix,
    ctx: Matrix,
}

impl DecoderLayerT {
    pub fn new(dim: usize, hidden: usize) -> Self {
        Self {
            self_attn: MultiHeadAttentionT::new(dim),
            enc_dec_attn: MultiHeadAttentionT::new(dim),
            ff: FeedForwardT::new(dim, hidden),
            h1: Matrix::zeros(0, 0),
            ctx: Matrix::zeros(0, 0),
        }
    }

    pub fn forward(&self, x: &Tensor, enc_out: &Tensor) -> Tensor {
        let h1 = self.self_attn.forward(x);
        let ctx = if h1.data.rows == enc_out.data.rows && h1.data.cols == enc_out.data.cols {
            Tensor::add(&h1, enc_out)
        } else {
            h1.clone()
        };
        let h2 = self.enc_dec_attn.forward(&ctx);
        self.ff.forward(&h2)
    }

    pub fn forward_train(&mut self, x: &Matrix, enc_out: &Matrix) -> Matrix {
        self.h1 = self.self_attn.forward_train(x);
        if self.h1.rows == enc_out.rows && self.h1.cols == enc_out.cols {
            self.ctx = Matrix::add(&self.h1, enc_out);
        } else {
            self.ctx = self.h1.clone();
        }
        let h2 = self.enc_dec_attn.forward_train(&self.ctx);
        self.ff.forward_train(&h2)
    }

    /// Backward pass returns gradient w.r.t. the layer input and w.r.t. the
    /// encoder context that was added to the decoder state.
    pub fn backward(&mut self, grad_out: &Matrix) -> (Matrix, Matrix) {
        let grad_ff = self.ff.backward(grad_out);
        let grad_ctx = self.enc_dec_attn.backward(&grad_ff);
        let grad_h1 = grad_ctx.clone();
        let grad_enc = if self.h1.rows == self.ctx.rows && self.h1.cols == self.ctx.cols {
            grad_ctx.clone()
        } else {
            Matrix::zeros(grad_ctx.rows, grad_ctx.cols)
        };
        let grad_in = self.self_attn.backward(&grad_h1);
        (grad_in, grad_enc)
    }

    pub fn zero_grad(&mut self) {
        self.self_attn.zero_grad();
        self.enc_dec_attn.zero_grad();
        self.ff.zero_grad();
    }

    pub fn adam_step(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) {
        self.self_attn
            .adam_step(lr, beta1, beta2, eps, weight_decay);
        self.enc_dec_attn
            .adam_step(lr, beta1, beta2, eps, weight_decay);
        self.ff
            .adam_step(lr, beta1, beta2, eps, weight_decay);
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let mut params = self.self_attn.parameters();
        params.extend(self.enc_dec_attn.parameters());
        params.extend(self.ff.parameters());
        params
    }
}

pub struct DecoderT {
    pub layers: Vec<DecoderLayerT>,
    pub embedding: EmbeddingT,
    pub proj: LinearT,
    enc_out_cache: Matrix,
}

impl DecoderT {
    pub fn new(n: usize, vocab_size: usize, model_dim: usize, hidden: usize) -> Self {
        let mut v = Vec::new();
        for _ in 0..n {
            v.push(DecoderLayerT::new(model_dim, hidden));
        }
        Self {
            layers: v,
            embedding: EmbeddingT::new(vocab_size, model_dim),
            proj: LinearT::new(model_dim, vocab_size),
            enc_out_cache: Matrix::zeros(0, 0),
        }
    }

    pub fn forward(&self, one_hot_x: &Tensor, enc_out: &Tensor) -> Tensor {
        // one_hot_x = (seq_len × vocab_size)
        let mut h = self.embedding.forward(one_hot_x);
        for l in &self.layers {
            h = l.forward(&h, enc_out);
        }
        self.proj.forward(&h)
    }

    pub fn forward_train(&mut self, one_hot_x: &Matrix, enc_out: &Matrix) -> Matrix {
        let mut h = self.embedding.forward_train(one_hot_x);
        for l in self.layers.iter_mut() {
            h = l.forward_train(&h, enc_out);
        }
        self.enc_out_cache = enc_out.clone();
        self.proj.forward_train(&h)
    }

    /// Backward pass through the decoder.  Returns the gradient with respect to
    /// the encoder output so that it can be backpropagated through the encoder
    /// as well.
    pub fn backward(&mut self, grad_out: &Matrix) -> Matrix {
        let mut g = self.proj.backward(grad_out);
        let mut grad_enc = Matrix::zeros(self.enc_out_cache.rows, self.enc_out_cache.cols);
        for l in self.layers.iter_mut().rev() {
            let (ng, genc) = l.backward(&g);
            g = ng;
            grad_enc = Matrix::add(&grad_enc, &genc);
        }
        self.embedding.backward(&g);
        grad_enc
    }

    pub fn zero_grad(&mut self) {
        self.embedding.zero_grad();
        self.proj.zero_grad();
        for l in self.layers.iter_mut() {
            l.zero_grad();
        }
    }

    pub fn adam_step(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) {
        self.embedding
            .adam_step(lr, beta1, beta2, eps, weight_decay);
        self.proj
            .adam_step(lr, beta1, beta2, eps, weight_decay);
        for l in self.layers.iter_mut() {
            l.adam_step(lr, beta1, beta2, eps, weight_decay);
        }
    }

    pub fn parameters(&mut self) -> Vec<&mut LinearT> {
        let (embedding, proj) = (&mut self.embedding, &mut self.proj);
        let mut params = embedding.parameters();
        params.push(proj);
        for l in self.layers.iter_mut() {
            params.extend(l.parameters());
        }
        params
    }
}
