use crate::math;
use crate::math::Matrix;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

/// N-dimensional tensor backed by a flat `Vec<f32>`.
///
/// The tensor stores its shape explicitly allowing operations on
/// higher-rank data.  Many legacy parts of the codebase still operate on
/// the 2-D [`Matrix`] type, so conversion helpers are provided to ease the
/// transition.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// Tensor elements in row-major order.
    pub data: Vec<f32>,
    /// Sizes for each dimension.
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor from raw parts.  The number of elements in `data`
    /// must match the product of the requested `shape`.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Tensor { data, shape }
    }

    /// Convenience constructor taking ownership of a [`Matrix`] while
    /// recording its two dimensional shape.  This is primarily used while
    /// refactoring existing code that still produces matrices.
    pub fn from_matrix(m: Matrix) -> Self {
        Tensor {
            shape: vec![m.rows, m.cols],
            data: m.data,
        }
    }

    /// Compute the flat index for a multi-dimensional coordinate.
    fn offset(&self, idx: &[usize]) -> usize {
        assert_eq!(idx.len(), self.shape.len());
        let mut stride = 1;
        let mut off = 0usize;
        for (i, &dim) in self.shape.iter().rev().enumerate() {
            let id = idx[self.shape.len() - 1 - i];
            assert!(id < dim, "index out of bounds");
            off += id * stride;
            stride *= dim;
        }
        off
    }

    /// Basic immutable indexing.
    pub fn get(&self, idx: &[usize]) -> f32 {
        let off = self.offset(idx);
        self.data[off]
    }

    /// Mutable indexing support.
    pub fn set(&mut self, idx: &[usize], value: f32) {
        let off = self.offset(idx);
        self.data[off] = value;
    }

    /// Change the view of the underlying data without modifying order.
    /// The new shape must contain the same number of elements.
    pub fn reshape(&mut self, new_shape: Vec<usize>) {
        assert_eq!(self.data.len(), new_shape.iter().product::<usize>());
        self.shape = new_shape;
    }

    /// Quantize the tensor to int8 values returning the packed data and
    /// a scale factor.  The scale maps the original floating point range to
    /// [-128, 127].
    pub fn quantize(&self) -> (Vec<i8>, f32) {
        let max = self.data.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let scale = if max == 0.0 { 1.0 } else { 127.0 / max };
        let q = self
            .data
            .iter()
            .map(|&v| {
                let scaled = v * scale;
                scaled.round().clamp(-128.0, 127.0) as i8
            })
            .collect();
        (q, scale)
    }

    /// Dequantize int8 data produced by [`quantize`].  The caller must provide
    /// the original shape which is not stored with the quantized values.
    pub fn dequantize(data: &[i8], scale: f32, shape: Vec<usize>) -> Tensor {
        let inv = if scale == 0.0 { 1.0 } else { 1.0 / scale };
        let deq = data.iter().map(|&v| v as f32 * inv).collect();
        Tensor { data: deq, shape }
    }

    /// Broadcast the tensor to a larger shape following numpy semantics
    /// where dimensions of size 1 can be expanded.
    pub fn broadcast_to(&self, target: &[usize]) -> Tensor {
        assert!(target.len() >= self.shape.len());
        for (&src, &dst) in self.shape.iter().rev().zip(target.iter().rev()) {
            assert!(src == dst || src == 1, "cannot broadcast dimension");
        }

        let out_len: usize = target.iter().product();
        let mut out = vec![0.0; out_len];

        // Prepare padded shapes and strides for easier index mapping.
        let mut src_shape = vec![1; target.len()];
        let offset = target.len() - self.shape.len();
        for (i, &dim) in self.shape.iter().enumerate() {
            src_shape[offset + i] = dim;
        }
        let mut src_stride = vec![0; target.len()];
        let mut stride = 1;
        for (i, dim) in src_shape.iter().rev().enumerate() {
            src_stride[src_shape.len() - 1 - i] = stride;
            stride *= *dim;
        }

        for i in 0..out_len {
            // Decode `i` into indices of the target shape and compute the
            // corresponding source index.
            let mut tmp = i;
            let mut src_index = 0usize;
            for ((&t_dim, &s_dim), &s_stride) in target
                .iter()
                .rev()
                .zip(src_shape.iter().rev())
                .zip(src_stride.iter().rev())
            {
                let idx = tmp % t_dim;
                tmp /= t_dim;
                let s_idx = if s_dim == 1 { 0 } else { idx };
                src_index += s_idx * s_stride;
            }
            out[i] = self.data[src_index];
        }

        Tensor {
            data: out,
            shape: target.to_vec(),
        }
    }

    /// Elementwise addition with broadcasting handled by the math helper.
    pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
        math::tensor_add(a, b)
    }

    /// Matrix multiplication treating the last two dimensions as matrices.
    pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
        math::tensor_matmul(a, b)
    }

    /// Transpose a 2-D tensor.
    pub fn transpose(t: &Tensor) -> Tensor {
        math::tensor_transpose(t)
    }

    /// Softmax along the last dimension.
    pub fn softmax(t: &Tensor) -> Tensor {
        math::tensor_softmax(t)
    }

    /// Softmax along the last dimension, writing the result into `out`.
    /// The input tensor `t` is not modified.
    pub fn softmax_into(out: &mut Tensor, t: &Tensor) {
        math::tensor_softmax_into(out, t)
    }

    /// In-place softmax along the last dimension.
    pub fn softmax_inplace(&mut self) {
        math::tensor_softmax_inplace(self)
    }

    /// Compute cross-entropy loss and gradient w.r.t. the logits contained in
    /// this tensor.  The returned gradient tensor can be fed directly into a
    /// backward pass of subsequent layers.
    #[allow(dead_code)]
    pub fn softmax_cross_entropy(&self, targets: &[usize], row_offset: usize) -> (f32, Tensor) {
        let (loss, grad) = math::tensor_softmax_cross_entropy(self, targets, row_offset);
        (loss, grad)
    }

    /// Create a tensor of zeros with the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; len],
            shape,
        }
    }

    /// Create a tensor of zeros matching the shape of `other`.
    pub fn zeros_like(other: &Tensor) -> Self {
        Tensor {
            data: vec![0.0; other.data.len()],
            shape: other.shape.clone(),
        }
    }

    /// Create a tensor of ones matching the shape of `other`.
    pub fn ones_like(other: &Tensor) -> Self {
        Tensor {
            data: vec![1.0; other.data.len()],
            shape: other.shape.clone(),
        }
    }

    /// Convert this tensor into an autograd [`Node`].
    pub fn into_node(self, requires_grad: bool) -> NodeRef {
        Node::new(self, requires_grad)
    }
}

/// Reference-counted node pointer used for building computation graphs.
pub type NodeRef = Rc<RefCell<Node>>;

/// Operations that a [`Node`] can represent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    None,
    Add,
    MatMul,
    Transpose,
    Softmax,
}

/// Autograd node storing the value, gradient and parent links.
#[derive(Clone, Debug)]
pub struct Node {
    /// Value of this node.
    pub value: Tensor,
    /// Gradient accumulated for this node.
    pub grad: Option<Tensor>,
    /// Operation that produced this node.
    op: Op,
    /// Parents contributing to this node.
    parents: Vec<NodeRef>,
    /// Whether this node requires gradients.
    pub requires_grad: bool,
}

impl Node {
    /// Create a leaf node from a tensor.
    pub fn new(value: Tensor, requires_grad: bool) -> NodeRef {
        Rc::new(RefCell::new(Node {
            value,
            grad: None,
            op: Op::None,
            parents: vec![],
            requires_grad,
        }))
    }

    /// Wrap a tensor that should participate in autograd.
    pub fn from_tensor(t: Tensor) -> NodeRef {
        Node::new(t, true)
    }

    /// Create a node without tracking gradients.
    pub fn from_tensor_no_grad(t: Tensor) -> NodeRef {
        Node::new(t, false)
    }

    /// Elementwise addition operation.
    pub fn add(a: &NodeRef, b: &NodeRef) -> NodeRef {
        let value = Tensor::add(&a.borrow().value, &b.borrow().value);
        let requires_grad = a.borrow().requires_grad || b.borrow().requires_grad;
        Rc::new(RefCell::new(Node {
            value,
            grad: None,
            op: Op::Add,
            parents: vec![a.clone(), b.clone()],
            requires_grad,
        }))
    }

    /// Matrix multiplication operation.
    pub fn matmul(a: &NodeRef, b: &NodeRef) -> NodeRef {
        let value = Tensor::matmul(&a.borrow().value, &b.borrow().value);
        let requires_grad = a.borrow().requires_grad || b.borrow().requires_grad;
        Rc::new(RefCell::new(Node {
            value,
            grad: None,
            op: Op::MatMul,
            parents: vec![a.clone(), b.clone()],
            requires_grad,
        }))
    }

    /// Transpose operation for 2-D tensors.
    pub fn transpose(x: &NodeRef) -> NodeRef {
        let value = Tensor::transpose(&x.borrow().value);
        Rc::new(RefCell::new(Node {
            value,
            grad: None,
            op: Op::Transpose,
            parents: vec![x.clone()],
            requires_grad: x.borrow().requires_grad,
        }))
    }

    /// Softmax along the last dimension.
    pub fn softmax(x: &NodeRef) -> NodeRef {
        let value = Tensor::softmax(&x.borrow().value);
        Rc::new(RefCell::new(Node {
            value,
            grad: None,
            op: Op::Softmax,
            parents: vec![x.clone()],
            requires_grad: x.borrow().requires_grad,
        }))
    }

    /// Trigger backward propagation starting from this node.
    pub fn backward(root: &NodeRef) {
        {
            let mut r = root.borrow_mut();
            if r.grad.is_none() {
                r.grad = Some(Tensor::ones_like(&r.value));
            }
        }
        let mut visited = HashSet::new();
        Node::backward_rec(root, &mut visited);
    }

    fn backward_rec(node: &NodeRef, visited: &mut HashSet<usize>) {
        let id = Rc::as_ptr(node) as usize;
        if visited.contains(&id) {
            return;
        }
        visited.insert(id);

        let grads = {
            let n = node.borrow();
            let grad_out = match &n.grad {
                Some(g) => g.clone(),
                None => return,
            };
            match n.op {
                Op::None => vec![],
                Op::Add => vec![grad_out.clone(), grad_out],
                Op::MatMul => {
                    let a = n.parents[0].borrow();
                    let b = n.parents[1].borrow();
                    let grad_a = Tensor::matmul(&grad_out, &Tensor::transpose(&b.value));
                    let grad_b = Tensor::matmul(&Tensor::transpose(&a.value), &grad_out);
                    vec![grad_a, grad_b]
                }
                Op::Transpose => vec![Tensor::transpose(&grad_out)],
                Op::Softmax => {
                    let s = &n.value;
                    let mut grad_in = Tensor::zeros_like(s);
                    let last_dim = *s.shape.last().unwrap();
                    let rows = s.data.len() / last_dim;
                    for r in 0..rows {
                        let offset = r * last_dim;
                        let mut dot = 0.0;
                        for i in 0..last_dim {
                            dot += grad_out.data[offset + i] * s.data[offset + i];
                        }
                        for i in 0..last_dim {
                            grad_in.data[offset + i] =
                                (grad_out.data[offset + i] - dot) * s.data[offset + i];
                        }
                    }
                    vec![grad_in]
                }
            }
        };

        for (parent, grad) in node.borrow().parents.iter().zip(grads.into_iter()) {
            if !parent.borrow().requires_grad {
                continue;
            }
            {
                let mut p = parent.borrow_mut();
                if let Some(existing) = &p.grad {
                    p.grad = Some(Tensor::add(existing, &grad));
                } else {
                    p.grad = Some(grad);
                }
            }
            Node::backward_rec(parent, visited);
        }
    }
}
