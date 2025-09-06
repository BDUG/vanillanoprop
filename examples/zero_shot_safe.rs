use std::collections::HashMap;
use vanillanoprop::config::Config;
use vanillanoprop::rl::zero_shot_safe::{SafeEnv, ZeroShotSafeAgent};
use vanillanoprop::rl::Env;

/// A tiny line environment with an unsafe region.
///
/// Moving left decreases the position; crossing the failure threshold triggers
/// `is_failure`.
struct SafeLineWorld {
    position: i32,
    fail_pos: f32,
}

impl Env for SafeLineWorld {
    type State = i32;
    type Action = i32;

    fn reset(&mut self) -> Self::State {
        self.position = 0;
        self.position
    }

    fn step(&mut self, action: Self::Action) -> (Self::State, f32) {
        self.position += action;
        (self.position, 0.0)
    }

    fn is_terminal(&self) -> bool {
        false
    }
}

impl SafeEnv for SafeLineWorld {
    fn is_failure(&self) -> bool {
        (self.position as f32) < self.fail_pos
    }
}

fn main() {
    // Load configuration and fall back to defaults if missing.
    let cfg = Config::from_path("configs/zero_shot_safe.toml").unwrap_or_default();
    println!("using gamma {}", cfg.gamma);

    // Create environment and agent guarding policy updates with a safety check.
    let env = SafeLineWorld {
        position: 0,
        fail_pos: -0.5,
    };
    let mut agent = ZeroShotSafeAgent::new(env);

    // Recovery policy moves right when a failure occurs.
    let mut recovery = HashMap::new();
    recovery.insert(-1, 1);
    agent.set_recovery_policy(recovery);

    // Safe update at the starting state.
    let s0 = agent.env.reset();
    agent.update_policy(s0, 1);

    // Move into the unsafe region and attempt another update.
    let (s1, _) = agent.env.step(-1);
    agent.update_policy(s1, -1);

    println!("Policy for state 0: {:?}", agent.policy.get(&0));
    println!("Policy for state -1: {:?}", agent.policy.get(&-1));
    println!("Safety violations: {}", agent.violations);
}
