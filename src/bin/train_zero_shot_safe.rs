use rand::Rng;
use serde::Deserialize;
use vanillanoprop::rl::zero_shot_safe::{SafeEnv, ZeroShotSafeAgent};
use vanillanoprop::rl::Env;

mod common;

#[derive(Deserialize)]
struct Config {
    discount_factor: f32,
    safety_thresholds: Vec<f32>,
    learning_rate: f32,
    rollout_steps: usize,
}

struct SafeLineWorld {
    position: i32,
    goal: i32,
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
        let reward = if self.position == self.goal {
            1.0
        } else {
            -0.01
        };
        (self.position, reward)
    }

    fn is_terminal(&self) -> bool {
        self.position == self.goal || self.position <= -self.goal
    }
}

impl SafeEnv for SafeLineWorld {
    fn is_failure(&self) -> bool {
        (self.position as f32) < self.fail_pos
    }
}

fn main() {
    let args = common::init_logging();
    let mut cfg_path = "configs/zero_shot_safe_config.toml".to_string();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--config" && i + 1 < args.len() {
            cfg_path = args[i + 1].clone();
            i += 2;
        } else {
            i += 1;
        }
    }
    let cfg_str = std::fs::read_to_string(&cfg_path).expect("read config");
    let cfg: Config = toml::from_str(&cfg_str).expect("parse config");
    vanillanoprop::info!(
        "discount {} lr {} rollout {}",
        cfg.discount_factor,
        cfg.learning_rate,
        cfg.rollout_steps
    );
    let env = SafeLineWorld {
        position: 0,
        goal: 5,
        fail_pos: cfg.safety_thresholds.get(0).copied().unwrap_or(-5.0),
    };
    let mut agent = ZeroShotSafeAgent::new(env);
    let actions = [-1, 1];
    let mut rng = rand::thread_rng();
    let episodes = 10;

    for ep in 0..episodes {
        let mut state = agent.env.reset();
        for _ in 0..cfg.rollout_steps {
            let action = actions[rng.gen_range(0..actions.len())];
            let (next_state, _reward) = agent.env.step(action);
            agent.update_policy(state, action);
            state = next_state;
            if agent.env.is_terminal() {
                break;
            }
        }
        vanillanoprop::info!(
            "Episode {} complete, violations {}",
            ep + 1,
            agent.violations
        );
    }
}
