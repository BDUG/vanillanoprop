use super::dataloader::Dataset;

/// Minimal audio dataset backed by vectors of samples.
///
/// The dataset contains synthetic waveforms and labels and is primarily used
/// for demonstrating the [`Dataset`] trait implementation.
pub struct AudioDataset;

impl Dataset for AudioDataset {
    type Item = (Vec<f32>, usize);

    fn load() -> Vec<Self::Item> {
        vec![
            (vec![0.0; 16000], 0),
            (vec![1.0; 16000], 1),
        ]
    }
}
