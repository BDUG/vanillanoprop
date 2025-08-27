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
    pub children: HashMap<A, TreeNode<S, A>>,
}

impl<S: Clone, A: Eq + Hash + Clone> TreeNode<S, A> {
    fn new(state: S) -> Self {
        Self {
            state,
            value: 0.0,
            visits: 0,
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
}

impl<E: Env> TreePoAgent<E> {
    /// Create a new agent with the given environment and hyperparameters.
    pub fn new(mut env: E, gamma: f32, lam: f32, max_depth: usize, rollout_steps: usize) -> Self {
        let state = env.reset();
        Self {
            env,
            root: TreeNode::new(state),
            gamma,
            lam,
            max_depth,
            rollout_steps,
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
        node.children.get_mut(&action).unwrap()
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

    /// Update the root policy based on child advantages. This is a simple
    /// normalised advantage scheme used as a placeholder for the policy update
    /// described in the TreePO paper.
    pub fn update_policy(&mut self) {
        let total: f32 = self
            .root
            .children
            .values()
            .map(|c| Self::advantage(c).max(0.0))
            .sum();
        if total > 0.0 {
            for child in self.root.children.values_mut() {
                let adv = Self::advantage(child).max(0.0);
                child.value = adv / total;
            }
        }
    }
}
