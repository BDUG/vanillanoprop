use std::collections::HashMap;

use crate::layers::LinearT;
use crate::math::Matrix;
use crate::metrics;
use crate::optim::{Loss, Optimizer};

/// A simple directed graph representing a neural network together with
/// minimal training utilities.
///
/// Nodes are stored as strings identifying the layer type or name. Edges
/// represent data flow between nodes. This structure allows models to be
/// composed programmatically by adding nodes and connecting them.  The
/// optional optimizer and loss are used by the `fit` and `evaluate`
/// helpers to provide a lightweight training API for examples.
pub struct Model {
    /// List of node labels in insertion order.
    pub nodes: Vec<String>,
    /// Directed edges between nodes represented by their indices.
    pub edges: Vec<(usize, usize)>,
    /// Optional metadata for nodes.
    pub metadata: HashMap<usize, HashMap<String, String>>,
    optimizer: Option<Box<dyn Optimizer>>, // training optimizer
    loss: Option<Box<dyn Loss>>,           // loss function
}

impl Default for Model {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            metadata: HashMap::new(),
            optimizer: None,
            loss: None,
        }
    }
}

impl Model {
    /// Create an empty model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure the model for training by providing an optimizer and loss
    /// function.
    pub fn compile<O, L>(&mut self, optimizer: O, loss: L)
    where
        O: Optimizer + 'static,
        L: Loss + 'static,
    {
        self.optimizer = Some(Box::new(optimizer));
        self.loss = Some(Box::new(loss));
    }

    /// Perform a single optimization step on the provided parameters.
    pub fn fit(&mut self, params: &mut [&mut LinearT]) {
        if let Some(opt) = &mut self.optimizer {
            opt.step(params);
        }
    }

    /// Mutable access to the underlying optimizer for advanced configuration.
    pub fn optimizer_mut(&mut self) -> Option<&mut dyn Optimizer> {
        self.optimizer.as_deref_mut()
    }

    /// Optimization step for models exposing raw matrices and biases.
    pub fn fit_fc(&mut self, fc: &mut Matrix, bias: &mut [f32], grad: &[f32], feat: &[f32]) {
        if let Some(opt) = &mut self.optimizer {
            opt.update_fc(fc, bias, grad, feat);
        }
    }

    /// Evaluate predictions against targets using the F1 score metric.
    pub fn evaluate(&self, pred: &[usize], tgt: &[usize]) -> f32 {
        metrics::f1_score(pred, tgt)
    }

    /// Predict the most likely class from a slice of logits.
    pub fn predict(&self, logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Add a node to the model and return its index.
    pub fn add<N: Into<String>>(&mut self, name: N) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(name.into());
        idx
    }

    /// Connect two nodes in the directed graph.
    ///
    /// `from` and `to` are node indices previously returned by `add`.
    pub fn connect(&mut self, from: usize, to: usize) {
        self.edges.push((from, to));
    }
}
