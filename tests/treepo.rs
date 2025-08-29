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
    let mut agent = TreePoAgent::new(env, 1.0, 1.0, 1, 1, 1.0);

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

    let adv0 = TreePoAgent::<TwoStateEnv>::advantage(
        agent
            .root
            .children
            .get(&0)
            .expect("child node for action 0 missing"),
    );
    let adv1 = TreePoAgent::<TwoStateEnv>::advantage(
        agent
            .root
            .children
            .get(&1)
            .expect("child node for action 1 missing"),
    );
    assert!(adv0 > adv1, "expansion should favour optimal action");
}

#[test]
fn policy_update_improves_over_random() {
    let env = TwoStateEnv::new();
    let mut agent = TreePoAgent::new(env, 1.0, 1.0, 1, 1, 1.0);

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

    let p_best = agent
        .root
        .children
        .get(&0)
        .expect("child node for action 0 missing")
        .policy;
    let expected_return: f32 = agent
        .root
        .children
        .iter()
        .map(|(a, child)| {
            let reward = if *a == 0 { 1.0 } else { 0.0 };
            child.policy * reward
        })
        .sum();

    // Random policy would yield 0.5 expected reward.
    assert!(p_best > 0.5, "policy should favour the better action");
    assert!(
        expected_return > 0.5,
        "policy update should improve expected return"
    );
}

#[derive(Clone)]
struct TwoStepEnv {
    state: i32,
}

impl TwoStepEnv {
    fn new() -> Self {
        Self { state: 0 }
    }
}

impl Env for TwoStepEnv {
    type State = i32;
    type Action = i32;

    fn reset(&mut self) -> Self::State {
        self.state = 0;
        self.state
    }

    fn step(&mut self, action: Self::Action) -> (Self::State, f32) {
        match self.state {
            0 => {
                if action == 0 {
                    self.state = 2;
                    (self.state, 1.0)
                } else {
                    self.state = 1;
                    (self.state, 0.0)
                }
            }
            1 => {
                self.state = 2;
                (self.state, 2.0)
            }
            _ => (self.state, 0.0),
        }
    }

    fn is_terminal(&self) -> bool {
        self.state == 2
    }
}

#[test]
fn policy_update_backpropagates_advantage() {
    let env = TwoStepEnv::new();
    let mut agent = TreePoAgent::new(env, 1.0, 1.0, 2, 2, 1.0);

    // Expand immediate reward branch.
    agent.env.reset();
    let (next0, r0) = agent.env.step(0);
    let mut root = agent.root.clone();
    let child0 = agent.expand_node(&mut root, 0, next0);
    TreePoAgent::<TwoStepEnv>::backup(child0, r0);

    // Expand delayed reward branch.
    agent.env.reset();
    let (mid, _) = agent.env.step(1);
    let child1 = agent.expand_node(&mut root, 1, mid);
    // From mid state take any action to receive reward.
    let (terminal, r2) = agent.env.step(0);
    let grand_child = agent.expand_node(child1, 0, terminal);
    TreePoAgent::<TwoStepEnv>::backup(grand_child, r2);
    agent.root = root;

    agent.update_policy();

    let p0 = agent
        .root
        .children
        .get(&0)
        .expect("child for action 0 missing")
        .policy;
    let p1 = agent
        .root
        .children
        .get(&1)
        .expect("child for action 1 missing")
        .policy;

    assert!(p1 > p0, "policy should favour delayed higher reward");
}
