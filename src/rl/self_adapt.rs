use super::{language_env::LanguageEnv, treepo::Env};
use crate::math::{argmax, softmax_cross_entropy, Matrix};
use crate::models::{DecoderT, TransformerEncoder};
use crate::reward::RewardModel;

/// Agent that adapts itself online by updating a Transformer
/// encoder/decoder on the reward signal from a [`LanguageEnv`].
///
/// At each step the current environment state is encoded, the decoder
/// predicts the next token, the environment returns a reward and the
/// true token. The agent then performs a backward pass and updates the
/// model parameters using a simple Adam step.
pub struct SelfAdaptAgent<R: RewardModel> {
    pub env: LanguageEnv,
    pub encoder: TransformerEncoder,
    pub decoder: DecoderT,
    pub lr: f32,
    pub vocab_size: usize,
    reward_model: R,
    state: Vec<u8>,
}

impl<R: RewardModel> SelfAdaptAgent<R> {
    /// Construct a new agent with the given environment and models.
    pub fn new(
        mut env: LanguageEnv,
        encoder: TransformerEncoder,
        decoder: DecoderT,
        lr: f32,
        vocab_size: usize,
        reward_model: R,
    ) -> Self {
        let state = env.reset();
        Self {
            env,
            encoder,
            decoder,
            lr,
            vocab_size,
            reward_model,
            state,
        }
    }

    fn one_hot(&self, tokens: &[u8]) -> Matrix {
        let seq_len = tokens.len().max(1);
        let mut m = Matrix::zeros(seq_len, self.vocab_size);
        for (i, &tok) in tokens.iter().enumerate() {
            m.set(i, tok as usize, 1.0);
        }
        m
    }

    /// Perform a single environment step, update the models and return the reward.
    pub fn step(&mut self) -> Option<f32> {
        if self.env.is_terminal() {
            return None;
        }
        let input = self.one_hot(&self.state);
        self.encoder.zero_grad();
        self.decoder.zero_grad();
        let enc_out = self.encoder.forward_train(&input, None);
        let logits = self.decoder.forward_train(&input, &enc_out);
        let last_row = logits.rows - 1;
        let row_slice = &logits.data[last_row * logits.cols..(last_row + 1) * logits.cols];
        let action = argmax(row_slice) as u8;
        let (next_state, _env_reward) = self.env.step(action);
        let expected = *next_state.last().unwrap();
        let reward = self.reward_model.score(&[action], &[expected]);
        let target = expected as usize;
        let (_loss, grad, _preds) = softmax_cross_entropy(&logits, &[target], last_row);
        let grad_enc = self.decoder.backward(&grad);
        self.encoder.backward(&grad_enc);
        self.decoder.adam_step(self.lr, 0.9, 0.999, 1e-8, 0.0);
        self.encoder.adam_step(self.lr, 0.9, 0.999, 1e-8, 0.0);
        self.state = next_state;
        Some(reward)
    }

    /// Reset the environment and clear internal state.
    pub fn reset(&mut self) {
        self.state = self.env.reset();
    }
}
