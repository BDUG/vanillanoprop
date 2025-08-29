use rand::Rng;
use std::collections::HashMap;
use vanillanoprop::rl::treepo::TreeNode;
use vanillanoprop::rl::{Env, TreePoAgent};

/// A toy 1-D environment where the agent moves left or right on a line.
/// The goal is to reach `goal`; stepping beyond `-goal` ends the episode
/// with a small penalty. Reaching the goal yields a reward of +1.
struct LineWorld {
    position: i32,
    goal: i32,
}

impl Env for LineWorld {
    type State = i32;
    type Action = i32;

    fn reset(&mut self) -> Self::State {
        // Start in the middle of the line for each episode
        self.position = 0;
        self.position
    }

    fn step(&mut self, action: Self::Action) -> (Self::State, f32) {
        // Move left (-1) or right (+1)
        self.position += action;
        let reward = if self.position == self.goal {
            // Provide a strong reward when the goal is reached
            1.0
        } else {
            // Small negative reward encourages shorter paths
            -0.01
        };
        (self.position, reward)
    }

    fn is_terminal(&self) -> bool {
        // Episode terminates when we reach the goal or go too far the other way
        self.position == self.goal || self.position <= -self.goal
    }
}

fn main() {
    // Create the environment and the TreePo agent.
    // Hyperparameters:
    // - gamma: discount factor for future rewards (0.9).
    // - lam: GAE smoothing factor (0.95).
    // - max_depth: maximum depth of the search tree (10).
    // - rollout_steps: number of steps to simulate during rollouts (10).
    // Try tweaking these values to see how learning changes.
    let env = LineWorld {
        position: 0,
        goal: 5,
    };
    let mut agent = TreePoAgent::new(env, 0.9, 0.95, 10, 10, 0.1);

    // Actions available to the agent: move left or right
    let actions = [-1, 1];
    let mut rng = rand::thread_rng();
    let episodes = 5; // Increase for longer training

    for episode in 0..episodes {
        // Reset environment and root node at the start of each episode
        let state = agent.env.reset();
        agent.root = TreeNode {
            state,
            value: 0.0,
            visits: 0,
            policy: 1.0,
            children: HashMap::new(),
        };

        // Collect data by interacting with the environment
        loop {
            // Sample a random action; a real agent would use a policy here
            let action = actions[rng.gen_range(0..actions.len())];
            let (next_state, reward) = agent.env.step(action);

            // Clone the root so we can mutably borrow it alongside the agent
            let mut root = agent.root.clone();
            // Expand the tree with the observed transition and backup reward
            let child = agent.expand_node(&mut root, action, next_state);
            TreePoAgent::<LineWorld>::backup(child, reward);
            // Write the updated root back to the agent
            agent.root = root;

            // Update the policy at the root based on accumulated rewards
            agent.update_policy();

            if agent.env.is_terminal() {
                break;
            }
        }

        // Inspect the learned probability of moving right from the root
        let p_right = agent.root.children.get(&1).map(|c| c.policy).unwrap_or(0.0);
        println!("Episode {}: P(move right)={:.2}", episode + 1, p_right);
    }
}
