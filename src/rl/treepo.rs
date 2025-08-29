use std::collections::HashMap;
use std::hash::Hash;

/// Trait representing a reinforcement learning environment.
///
/// The environment exposes a state and accepts actions. After each action the
/// environment returns the next state and a scalar reward. The environment
/// terminates when [`is_terminal`] returns `true`.
pub trait Env {
    /// Type used for representing states.
    type State: Clone;
    /// Type used for representing actions.
    type Action: Clone + Eq + Hash;

    /// Reset the environment to its initial state and return that state.
    fn reset(&mut self) -> Self::State;
    /// Advance the environment by taking an action. Returns the next state and
    /// a reward signal.
    fn step(&mut self, action: Self::Action) -> (Self::State, f32);
    /// Check whether the current state is terminal.
    fn is_terminal(&self) -> bool;
}

/// A node in the policy tree. Each node stores the state it represents, a
/// running value estimate, the visit count, and child nodes indexed by action.
#[derive(Clone)]
pub struct TreeNode<S, A> {
    pub state: S,
    pub value: f32,
    pub visits: u32,
    /// Probability of selecting the action leading to this node.
    pub policy: f32,
    pub children: HashMap<A, TreeNode<S, A>>,
}

impl<S: Clone, A: Eq + Hash + Clone> TreeNode<S, A> {
    fn new(state: S) -> Self {
        Self {
            state,
            value: 0.0,
            visits: 0,
            policy: 0.0,
            children: HashMap::new(),
        }
    }
}

/// TreePO agent maintaining a policy tree and value estimates.
///
/// The implementation here provides a light‑weight, generic skeleton of the
/// algorithm described in the Tree Policy Optimization paper. It keeps track of
/// node visit counts and value estimates, supports expanding new nodes,
/// computing advantages, backing values up the tree, and applying a very simple
/// policy update based on the estimated advantages.
pub struct TreePoAgent<E: Env> {
    pub env: E,
    pub root: TreeNode<E::State, E::Action>,
    pub gamma: f32,
    pub lam: f32,
    pub max_depth: usize,
    pub rollout_steps: usize,
    pub lr: f32,
}

impl<E: Env> TreePoAgent<E> {
    /// Create a new agent with the given environment and hyperparameters.
    pub fn new(
        mut env: E,
        gamma: f32,
        lam: f32,
        max_depth: usize,
        rollout_steps: usize,
        lr: f32,
    ) -> Self {
        let state = env.reset();
        let mut root = TreeNode::new(state);
        root.policy = 1.0;
        Self {
            env,
            root,
            gamma,
            lam,
            max_depth,
            rollout_steps,
            lr,
        }
    }

    /// Expand the given node by adding a child for `action` leading to
    /// `next_state`.
    pub fn expand_node<'a>(
        &mut self,
        node: &'a mut TreeNode<E::State, E::Action>,
        action: E::Action,
        next_state: E::State,
    ) -> &'a mut TreeNode<E::State, E::Action> {
        node.children
            .entry(action.clone())
            .or_insert_with(|| TreeNode::new(next_state));
        node
            .children
            .get_mut(&action)
            .expect("child node should exist after insertion")
    }

    /// Backup a value estimate through the node.
    pub fn backup(node: &mut TreeNode<E::State, E::Action>, reward: f32) {
        node.value += reward;
        node.visits += 1;
    }

    /// Compute the advantage (value estimate per visit) of a node.
    pub fn advantage(node: &TreeNode<E::State, E::Action>) -> f32 {
        if node.visits == 0 {
            0.0
        } else {
            node.value / node.visits as f32
        }
    }

    fn propagate_advantage(node: &TreeNode<E::State, E::Action>, gamma: f32) -> f32 {
        let mut adv = Self::advantage(node);
        if !node.children.is_empty() {
            let best_child = node
                .children
                .values()
                .map(|c| Self::propagate_advantage(c, gamma))
                .fold(f32::NEG_INFINITY, f32::max);
            if best_child.is_finite() {
                adv += gamma * best_child;
            }
        }
        adv
    }

    /// Update the root policy based on backpropagated advantages. Advantages are
    /// propagated from the leaves up the tree using discounted returns and then
    /// applied to the root policy via a simple gradient step followed by
    /// normalisation.
    pub fn update_policy(&mut self) {
        let mut advs: Vec<(E::Action, f32)> = Vec::new();
        for (action, child) in self.root.children.iter() {
            let adv = Self::propagate_advantage(child, self.gamma);
            advs.push((action.clone(), adv));
        }
        if advs.is_empty() {
            return;
        }
        let baseline = advs.iter().map(|(_, a)| *a).sum::<f32>() / advs.len() as f32;
        for (action, adv) in advs {
            if let Some(child) = self.root.children.get_mut(&action) {
                child.policy += self.lr * (adv - baseline);
                if child.policy < 0.0 {
                    child.policy = 0.0;
                }
            }
        }
        let sum: f32 = self.root.children.values().map(|c| c.policy).sum();
        if sum > 0.0 {
            for child in self.root.children.values_mut() {
                child.policy /= sum;
            }
        } else {
            let len = self.root.children.len() as f32;
            for child in self.root.children.values_mut() {
                child.policy = 1.0 / len;
            }
        }
    }
}
