use crate::huggingface::fetch_hf_files;
use crate::layers::LinearT;
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

/// Kinds of layers that expose trainable [`LinearT`] parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerKind {
    /// Fully connected / linear layers.
    Linear,
    /// Convolution layers.
    Conv,
}

/// Specification of a layer that should remain frozen during optimisation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FreezeSpec {
    /// The type of layer (e.g. linear, convolution).
    pub kind: LayerKind,
    /// Index of the parameter within the group of the specified `kind`.
    pub idx: usize,
}

/// Representation of a fine-tuning session.
///
/// Holds the set of layer indices that should remain frozen during
/// optimisation.  Call [`FineTune::filter`] with the full parameter list to
/// obtain only the unfrozen parameters for an update step.
#[derive(Debug, Clone)]
pub struct FineTune {
    frozen: Vec<FreezeSpec>,
}

impl FineTune {
    /// Create a new [`FineTune`] configuration with the provided frozen
    /// specifications.
    pub fn new(frozen: Vec<FreezeSpec>) -> Self {
        Self { frozen }
    }

    /// Filter the provided parameters, returning only those that are not
    /// frozen.
    ///
    /// The `params` vector should contain tuples of the layer kind and a
    /// mutable reference to the underlying [`LinearT`] parameters. Indices are
    /// tracked per layer kind, allowing heterogeneous parameter lists to be
    /// filtered correctly.
    pub fn filter<'a>(&self, params: Vec<(LayerKind, &'a mut LinearT)>) -> Vec<&'a mut LinearT> {
        let mut counts: HashMap<LayerKind, usize> = HashMap::new();
        params
            .into_iter()
            .filter_map(|(kind, p)| {
                let idx = counts.entry(kind).or_insert(0);
                let current = *idx;
                *idx += 1;
                if self
                    .frozen
                    .iter()
                    .any(|f| f.kind == kind && f.idx == current)
                {
                    None
                } else {
                    Some(p)
                }
            })
            .collect()
    }
}

/// Run the fine-tuning setup.
///
/// `model_id` is a Hugging Face model identifier.  The pre-trained weights are
/// fetched using [`fetch_hf_files`] and then loaded into the model by the
/// caller-provided `load_fn`.
///
/// `freeze_layers` specifies indices of parameters that should remain frozen
/// during optimisation.
///
/// Returns a [`FineTune`] helper which can be used to filter parameters prior
/// to optimisation steps.
pub fn run<F>(
    model_id: &str,
    freeze_layers: Vec<FreezeSpec>,
    mut load_fn: F,
) -> Result<FineTune, Box<dyn Error>>
where
    F: FnMut(&Path, &Path) -> Result<(), Box<dyn Error>>,
{
    let files = fetch_hf_files(model_id, None)?;
    load_fn(&files.config, &files.weights)?;
    Ok(FineTune::new(freeze_layers))
}

/// Convenience helper to parse a comma separated list of layer specifications
/// into a vector. Each entry should be of the form `"<kind>:<index>"`, e.g.
/// `"linear:0,conv:1"`.
pub fn parse_freeze_list(list: &str) -> Vec<FreezeSpec> {
    list.split(',')
        .filter_map(|s| {
            let mut parts = s.split(':');
            let kind = match parts.next()?.trim().to_lowercase().as_str() {
                "linear" => LayerKind::Linear,
                "conv" | "conv2d" => LayerKind::Conv,
                _ => return None,
            };
            let idx = parts.next()?.trim().parse::<usize>().ok()?;
            Some(FreezeSpec { kind, idx })
        })
        .collect()
}
