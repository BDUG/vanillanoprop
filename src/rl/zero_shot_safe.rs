use std::collections::HashMap;
use std::hash::Hash;

use super::Env;

/// Extension of [`Env`] that exposes a safety predicate.
///
/// A safe environment indicates whether the current state violates some
/// constraint via [`is_failure`]. Agents can use this information to avoid
/// updating their policy when unsafe states are encountered and optionally
/// trigger a recovery strategy.
pub trait SafeEnv: Env {
    /// Returns `true` when the environment is in a failure state.
    fn is_failure(&self) -> bool;
}

/// Minimal agent that guards policy updates with a safety check.
///
/// The agent maintains a policy mapping states to actions. Before applying any
/// update it queries the environment's [`SafeEnv::is_failure`] predicate. When a
/// failure is reported the update is skipped and, if available, a recovery
/// policy is used instead. The number of safety violations encountered so far is
/// tracked in [`violations`].
pub struct ZeroShotSafeAgent<E>
where
    E: SafeEnv,
    E::State: Eq + Hash,
{
    pub env: E,
    /// Current policy mapping states to actions.
    pub policy: HashMap<E::State, E::Action>,
    /// Optional recovery policy consulted when a failure occurs.
    pub recovery_policy: Option<HashMap<E::State, E::Action>>,
    /// Counter of how often the safety constraint was violated.
    pub violations: u32,
}

impl<E> ZeroShotSafeAgent<E>
where
    E: SafeEnv,
    E::State: Eq + Hash,
{
    /// Create a new agent with an empty policy and no recovery behaviour.
    pub fn new(env: E) -> Self {
        Self {
            env,
            policy: HashMap::new(),
            recovery_policy: None,
            violations: 0,
        }
    }

    /// Install a recovery policy that is consulted when a safety failure occurs.
    pub fn set_recovery_policy(&mut self, policy: HashMap<E::State, E::Action>) {
        self.recovery_policy = Some(policy);
    }

    /// Update the policy for `state` to `action`.
    ///
    /// The update is only applied when the environment reports no failure. When
    /// a failure is detected the update is skipped and the recovery policy is
    /// used instead, if one is available.
    pub fn update_policy(&mut self, state: E::State, action: E::Action) {
        if self.env.is_failure() {
            self.violations += 1;
            if let Some(recovery) = &self.recovery_policy {
                if let Some(rec_action) = recovery.get(&state).cloned() {
                    self.policy.insert(state, rec_action);
                }
            }
        } else {
            self.policy.insert(state, action);
        }
    }
}
