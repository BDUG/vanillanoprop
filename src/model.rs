use std::{collections::HashMap, fs, io};

use crate::layers::LinearT;
use crate::math::Matrix;
use crate::metrics;
use crate::optim::{Loss, Optimizer};
use crate::tensor::Tensor;
use crate::weights::{tensor_to_vec2, vec2_to_matrix};
use serde::{Deserialize, Serialize};

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
    /// Optional edges representing temporal flow between nodes.
    pub flow_edges: Vec<(usize, usize)>,
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
            flow_edges: Vec::new(),
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

    /// Optimization step for batched feature and gradient matrices.
    pub fn fit_fc_batch(
        &mut self,
        fc: &mut Matrix,
        bias: &mut [f32],
        grad: &Matrix,
        feat: &Matrix,
    ) {
        if let Some(opt) = &mut self.optimizer {
            for i in 0..grad.rows {
                let g_row = &grad.data[i * grad.cols..(i + 1) * grad.cols];
                let f_row = &feat.data[i * feat.cols..(i + 1) * feat.cols];
                opt.update_fc(fc, bias, g_row, f_row);
            }
        }
    }

    /// Evaluate predictions against targets using the F1 score metric.
    pub fn evaluate(&self, pred: &[usize], tgt: &[usize]) -> f32 {
        metrics::f1_score(pred, tgt)
    }

    /// Predict the most likely class from a slice of logits.
    pub fn predict(&self, logits: &[f32]) -> usize {
        use std::cmp::Ordering;

        if logits.is_empty() {
            return 0;
        }

        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
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

    /// Connect two nodes with a temporal flow edge.
    pub fn connect_flow(&mut self, from: usize, to: usize) {
        self.flow_edges.push((from, to));
    }

    /// Add a hybrid RNN -> Transformer block to the graph.
    ///
    /// `input` is the index of the preceding node. The function creates two
    /// new nodes labelled `rnn` and `transformer`, connects them in sequence
    /// and attaches the RNN to `input`. The index of the transformer node is
    /// returned allowing further composition.
    pub fn add_hybrid_block(&mut self, input: usize) -> usize {
        let rnn = self.add("rnn");
        let transformer = self.add("transformer");
        self.connect(input, rnn);
        self.connect(rnn, transformer);
        transformer
    }

    /// Annotate a node with a time attribute used for flow based models.
    pub fn set_time(&mut self, node: usize, time: f32) {
        let entry = self.metadata.entry(node).or_insert_with(HashMap::new);
        entry.insert("time".into(), time.to_string());
    }

    /// Persist the model architecture together with provided layer weights.
    ///
    /// The weights slice should correspond to the layers that make up this
    /// model in a consistent order.  Only the raw weight matrices are saved;
    /// optimiser and loss state are ignored.
    pub fn save(&self, path: &str, params: &[&LinearT]) -> Result<(), io::Error> {
        #[derive(Serialize, Deserialize)]
        struct ModelState {
            nodes: Vec<String>,
            edges: Vec<(usize, usize)>,
            flow_edges: Vec<(usize, usize)>,
            metadata: HashMap<usize, HashMap<String, String>>,
            weights: Vec<Vec<Vec<f32>>>,
        }

        let weights: Vec<Vec<Vec<f32>>> = params.iter().map(|p| tensor_to_vec2(&p.w)).collect();
        let state = ModelState {
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            flow_edges: self.flow_edges.clone(),
            metadata: self.metadata.clone(),
            weights,
        };
        let bin =
            bincode::serialize(&state).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        if let Some(parent) = std::path::Path::new(path).parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, bin)?;
        crate::info!("Saved model to {}", path);
        Ok(())
    }

    /// Load a model architecture and associated weights from disk.
    ///
    /// The provided `params` slice is updated in-place with the loaded
    /// weights.  Returns the reconstructed [`Model`] without any optimizer or
    /// loss configured.
    pub fn load(path: &str, params: &mut [&mut LinearT]) -> Result<Self, io::Error> {
        #[derive(Serialize, Deserialize)]
        struct ModelState {
            nodes: Vec<String>,
            edges: Vec<(usize, usize)>,
            flow_edges: Vec<(usize, usize)>,
            metadata: HashMap<usize, HashMap<String, String>>,
            weights: Vec<Vec<Vec<f32>>>,
        }

        let bin = fs::read(path)?;
        let state: ModelState =
            bincode::deserialize(&bin).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        for (p, w) in params.iter_mut().zip(state.weights.iter()) {
            p.w = Tensor::from_matrix(vec2_to_matrix(w));
        }
        crate::info!("Loaded model from {}", path);
        Ok(Model {
            nodes: state.nodes,
            edges: state.edges,
            flow_edges: state.flow_edges,
            metadata: state.metadata,
            optimizer: None,
            loss: None,
        })
    }
}
