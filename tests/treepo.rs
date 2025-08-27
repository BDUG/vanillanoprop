use vanillanoprop::rl::treepo::{Env, TreePoAgent};

#[derive(Clone)]
struct TwoStateEnv {
    state: i32,
}

impl TwoStateEnv {
    fn new() -> Self {
        Self { state: 0 }
    }
}

impl Env for TwoStateEnv {
    type State = i32;
    type Action = i32;

    fn reset(&mut self) -> Self::State {
        self.state = 0;
        self.state
    }

    fn step(&mut self, action: Self::Action) -> (Self::State, f32) {
        // Transition to terminal state with deterministic rewards.
        self.state = 1;
        if action == 0 {
            (self.state, 1.0)
        } else {
            (self.state, 0.0)
        }
    }

    fn is_terminal(&self) -> bool {
        self.state == 1
    }
}

#[test]
fn tree_expansion_selects_optimal_action() {
    let env = TwoStateEnv::new();
    let mut agent = TreePoAgent::new(env, 1.0);

    // Expand both actions from the root and backup their rewards.
    agent.env.reset();
    let (next0, r0) = agent.env.step(0);
    agent.env.reset();
    let (next1, r1) = agent.env.step(1);

    let mut root = agent.root.clone();
    let child0 = agent.expand_node(&mut root, 0, next0);
    TreePoAgent::<TwoStateEnv>::backup(child0, r0);
    let child1 = agent.expand_node(&mut root, 1, next1);
    TreePoAgent::<TwoStateEnv>::backup(child1, r1);
    agent.root = root;

    let adv0 = TreePoAgent::<TwoStateEnv>::advantage(agent.root.children.get(&0).unwrap());
    let adv1 = TreePoAgent::<TwoStateEnv>::advantage(agent.root.children.get(&1).unwrap());
    assert!(adv0 > adv1, "expansion should favour optimal action");
}

#[test]
fn policy_update_improves_over_random() {
    let env = TwoStateEnv::new();
    let mut agent = TreePoAgent::new(env, 1.0);

    // Collect data for both actions.
    agent.env.reset();
    let (next0, r0) = agent.env.step(0);
    agent.env.reset();
    let (next1, r1) = agent.env.step(1);

    let mut root = agent.root.clone();
    let child0 = agent.expand_node(&mut root, 0, next0);
    TreePoAgent::<TwoStateEnv>::backup(child0, r0);
    let child1 = agent.expand_node(&mut root, 1, next1);
    TreePoAgent::<TwoStateEnv>::backup(child1, r1);
    agent.root = root;

    // Update policy based on estimated advantages.
    agent.update_policy();

    let p_best = agent.root.children.get(&0).unwrap().value;
    let expected_return: f32 = agent
        .root
        .children
        .iter()
        .map(|(a, child)| {
            let reward = if *a == 0 { 1.0 } else { 0.0 };
            child.value * reward
        })
        .sum();

    // Random policy would yield 0.5 expected reward.
    assert!(p_best > 0.5, "policy should favour the better action");
    assert!(expected_return > 0.5, "policy update should improve expected return");
}

