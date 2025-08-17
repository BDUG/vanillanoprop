use crate::math::Matrix;

#[derive(Clone)]
pub enum Op {
    MatMul,
    Add,
}

#[derive(Clone)]
pub struct Node {
    pub op: Op,
    pub parents: Vec<Tensor>,
}

#[derive(Clone)]
pub struct Tensor {
    pub data: Matrix,
    pub grad: Option<Matrix>,
    pub requires_grad: bool,
    pub node: Option<Node>,
}

impl Tensor {
    pub fn from_matrix(data: Matrix, requires_grad: bool) -> Self {
        let grad = if requires_grad {
            Some(Matrix::zeros(data.rows, data.cols))
        } else {
            None
        };
        Tensor {
            data,
            grad,
            requires_grad,
            node: None,
        }
    }

    pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
        let data = a.data.add(&b.data);
        let requires_grad = a.requires_grad || b.requires_grad;
        let node = if requires_grad {
            Some(Node {
                op: Op::Add,
                parents: vec![a.clone(), b.clone()],
            })
        } else {
            None
        };
        Tensor {
            data,
            grad: None,
            requires_grad,
            node,
        }
    }

    pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
        let data = Matrix::matmul(&a.data, &b.data);
        let requires_grad = a.requires_grad || b.requires_grad;
        let node = if requires_grad {
            Some(Node {
                op: Op::MatMul,
                parents: vec![a.clone(), b.clone()],
            })
        } else {
            None
        };
        Tensor {
            data,
            grad: None,
            requires_grad,
            node,
        }
    }

    pub fn backward(&mut self) {
        if self.grad.is_none() {
            self.grad = Some(Matrix::from_vec(
                self.data.rows,
                self.data.cols,
                vec![1.0; self.data.rows * self.data.cols],
            ));
        }
        let mut stack = vec![self.clone()];
        while let Some(mut t) = stack.pop() {
            if let Some(ref node) = t.node {
                match node.op {
                    Op::Add => {
                        for parent in &node.parents {
                            if parent.requires_grad {
                                if let Some(ref mut pg) = parent.clone().grad.clone() {
                                    for i in 0..pg.data.len() {
                                        pg.data[i] += t.grad.as_ref().unwrap().data[i];
                                    }
                                }
                                stack.push(parent.clone());
                            }
                        }
                    }
                    Op::MatMul => {
                        let a = &node.parents[0];
                        let b = &node.parents[1];
                        let grad_out = t.grad.as_ref().unwrap();
                        if a.requires_grad {
                            let g_a =
                                Matrix::matmul(grad_out, &b.data.transpose());
                            if let Some(ref mut ag) = a.clone().grad.clone() {
                                for i in 0..ag.data.len() {
                                    ag.data[i] += g_a.data[i];
                                }
                            }
                            stack.push(a.clone());
                        }
                        if b.requires_grad {
                            let g_b =
                                Matrix::matmul(&a.data.transpose(), grad_out);
                            if let Some(ref mut bg) = b.clone().grad.clone() {
                                for i in 0..bg.data.len() {
                                    bg.data[i] += g_b.data[i];
                                }
                            }
                            stack.push(b.clone());
                        }
                    }
                }
            }
        }
    }
}
