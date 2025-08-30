use rand::Rng;
use vanillanoprop::rl::treepo::TreeNode;
use vanillanoprop::rl::{Env, TreePoAgent};

mod common;

struct LineWorld {
    position: i32,
    goal: i32,
}

impl Env for LineWorld {
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

fn main() {
    let _ = common::init_logging();
    let env = LineWorld {
        position: 0,
        goal: 5,
    };
    let mut agent = TreePoAgent::new(env, 0.9, 0.95, 10, 10, 0.1);
    let actions = [-1, 1];
    let mut rng = rand::thread_rng();
    let episodes = 10;

    for episode in 0..episodes {
        let state = agent.env.reset();
        agent.root.state = state;
        agent.root.value = 0.0;
        agent.root.visits = 0;
        agent.root.children.clear();
        loop {
            let action = actions[rng.gen_range(0..actions.len())];
            let (next_state, reward) = agent.env.step(action);
            {
                use std::collections::HashMap;
                let child = agent.root.children.entry(action).or_insert(TreeNode {
                    state: next_state,
                    value: 0.0,
                    visits: 0,
                    policy: 0.0,
                    children: HashMap::new(),
                });
                TreePoAgent::<LineWorld>::backup(child, reward);
            }
            agent.update_policy();
            if agent.env.is_terminal() {
                break;
            }
        }
        log::info!("Episode {} complete", episode + 1);
    }
}
