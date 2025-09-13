use vanillanoprop::layers::Activation;
use vanillanoprop::models::{DecoderT, TransformerEncoder};
use vanillanoprop::reward::NGramReward;
use vanillanoprop::rl::{Env, SelfAdaptAgent};

#[derive(Clone)]
struct EmptyStateEnv {
    step_called: bool,
}

impl EmptyStateEnv {
    fn new() -> Self {
        Self { step_called: false }
    }
}

impl Env for EmptyStateEnv {
    type State = Vec<u8>;
    type Action = u8;

    fn reset(&mut self) -> Self::State {
        self.step_called = false;
        Vec::new()
    }

    fn step(&mut self, _action: Self::Action) -> (Self::State, f32) {
        self.step_called = true;
        (Vec::new(), 0.0)
    }

    fn is_terminal(&self) -> bool {
        self.step_called
    }
}

#[test]
fn step_handles_empty_state() {
    let env = EmptyStateEnv::new();
    let encoder = TransformerEncoder::new(1, 2, 4, 1, 8, 0.0);
    let decoder = DecoderT::new(1, 2, 4, 8, Activation::ReLU, false, 1);
    let reward = NGramReward::new(1);
    let mut agent = SelfAdaptAgent::new(env, encoder, decoder, 1e-3, 2, reward);

    let reward = agent.step();
    assert!(reward.is_some(), "step should handle empty next state");
}
