use crate::huggingface::fetch_hf_files;
use crate::layers::LinearT;
use std::error::Error;
use std::path::Path;

/// Representation of a fine-tuning session.
///
/// Holds the set of layer indices that should remain frozen during
/// optimisation.  Call [`FineTune::filter`] with the full parameter list to
/// obtain only the unfrozen parameters for an update step.
#[derive(Debug, Clone)]
pub struct FineTune {
    frozen: Vec<usize>,
}

impl FineTune {
    /// Create a new [`FineTune`] configuration with the provided frozen layer
    /// indices.
    pub fn new(frozen: Vec<usize>) -> Self {
        Self { frozen }
    }

    /// Filter the provided parameters, returning only those that are not
    /// frozen.
    pub fn filter<'a>(&self, params: Vec<&'a mut LinearT>) -> Vec<&'a mut LinearT> {
        params
            .into_iter()
            .enumerate()
            .filter_map(|(i, p)| if self.frozen.contains(&i) { None } else { Some(p) })
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
    freeze_layers: Vec<usize>,
    mut load_fn: F,
) -> Result<FineTune, Box<dyn Error>>
where
    F: FnMut(&Path, &Path) -> Result<(), Box<dyn Error>>,
{
    let files = fetch_hf_files(model_id, None)?;
    load_fn(&files.config, &files.weights)?;
    Ok(FineTune::new(freeze_layers))
}

/// Convenience helper to parse a comma separated list of layer indices into a
/// vector.
pub fn parse_freeze_list(list: &str) -> Vec<usize> {
    list.split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect()
}
