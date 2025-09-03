use super::Env;

/// Environment that steps through a reference text token by token.
///
/// The state is the prefix of the reference text observed so far. At each
/// step the agent predicts the next token as an action. A reward of `1.0` is
/// returned when the prediction matches the next token in the reference text
/// and `0.0` otherwise.
pub struct LanguageEnv {
    reference: Vec<u8>,
    position: usize,
    state: Vec<u8>,
}

impl LanguageEnv {
    /// Create a new language environment from the given reference text.
    pub fn new(reference: Vec<u8>) -> Self {
        Self {
            reference,
            position: 0,
            state: Vec::new(),
        }
    }
}

impl Env for LanguageEnv {
    type State = Vec<u8>;
    type Action = u8;

    fn reset(&mut self) -> Self::State {
        self.position = 0;
        self.state.clear();
        self.state.clone()
    }

    fn step(&mut self, action: Self::Action) -> (Self::State, f32) {
        if self.is_terminal() {
            return (self.state.clone(), 0.0);
        }
        let expected = self.reference[self.position];
        let reward = if action == expected { 1.0 } else { 0.0 };
        self.state.push(expected);
        self.position += 1;
        (self.state.clone(), reward)
    }

    fn is_terminal(&self) -> bool {
        self.position >= self.reference.len()
    }
}
