pub trait RewardModel {
    /// Compute a reward for a prediction given the reference.
    /// Both arguments are token sequences.
    fn score(&self, prediction: &[u8], reference: &[u8]) -> f32;
}

/// Simple n-gram overlap reward.
///
/// Returns `1.0` when the last `n` tokens of the prediction and
/// reference match and `0.0` otherwise.
pub struct NGramReward {
    n: usize,
}

impl NGramReward {
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl RewardModel for NGramReward {
    fn score(&self, prediction: &[u8], reference: &[u8]) -> f32 {
        let n = self.n.min(prediction.len()).min(reference.len());
        if n == 0 {
            return 0.0;
        }
        let p_slice = &prediction[prediction.len() - n..];
        let r_slice = &reference[reference.len() - n..];
        if p_slice == r_slice { 1.0 } else { 0.0 }
    }
}

/// A reward model backed by an external scoring function.
/// Useful for plugging in custom reward hooks.
pub struct ExternalReward<F>(pub F)
where
    F: Fn(&[u8], &[u8]) -> f32;

impl<F> RewardModel for ExternalReward<F>
where
    F: Fn(&[u8], &[u8]) -> f32,
{
    fn score(&self, prediction: &[u8], reference: &[u8]) -> f32 {
        (self.0)(prediction, reference)
    }
}
