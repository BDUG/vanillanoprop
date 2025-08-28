use std::collections::HashMap;

/// A simple directed graph representing a neural network.
///
/// Nodes are stored as strings identifying the layer type or name.
/// Edges represent data flow between nodes. This structure allows
/// models to be composed programmatically by adding nodes and
/// connecting them.
#[derive(Clone, Default, Debug)]
pub struct Model {
    /// List of node labels in insertion order.
    pub nodes: Vec<String>,
    /// Directed edges between nodes represented by their indices.
    pub edges: Vec<(usize, usize)>,
    /// Optional metadata for nodes.
    pub metadata: HashMap<usize, HashMap<String, String>>,
}

impl Model {
    /// Create an empty model.
    pub fn new() -> Self {
        Self::default()
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

