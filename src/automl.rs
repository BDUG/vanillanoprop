use crate::config::Config;
use crate::logging::Logger;
use rand::{seq::SliceRandom, Rng};
use serde::Serialize;

/// Describes the hyperparameter options to explore during search.
#[derive(Debug, Clone)]
pub struct SearchSpace {
    /// Candidate learning rates.
    pub learning_rate: Vec<f32>,
    /// Possible batch sizes to evaluate.
    pub batch_size: Vec<usize>,
    /// Number of epochs for training.
    pub epochs: Vec<usize>,
    /// Discount factor for RL agents.
    pub gamma: Vec<f32>,
    /// Lambda parameter for advantage estimation.
    pub lam: Vec<f32>,
    /// Maximum search depth for tree based methods.
    pub max_depth: Vec<usize>,
    /// Number of rollout steps to use during planning.
    pub rollout_steps: Vec<usize>,
}

impl SearchSpace {
    /// Build a [`SearchSpace`] from a [`Config`]. Any field that contains
    /// multiple values will be expanded into the search space. Missing entries
    /// fall back to the configuration's current value.
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            learning_rate: if cfg.learning_rate.is_empty() {
                vec![0.001]
            } else {
                cfg.learning_rate.clone()
            },
            batch_size: vec![cfg.batch_size],
            epochs: vec![cfg.epochs],
            gamma: vec![cfg.gamma],
            lam: vec![cfg.lam],
            max_depth: vec![cfg.max_depth],
            rollout_steps: vec![cfg.rollout_steps],
        }
    }
}

#[derive(Serialize)]
struct AutoMlRecord {
    trial: usize,
    learning_rate: f32,
    batch_size: usize,
    epochs: usize,
    gamma: f32,
    lam: f32,
    max_depth: usize,
    rollout_steps: usize,
    score: f32,
    kind: &'static str,
}

/// Perform a grid search over all combinations in the [`SearchSpace`]. The
/// evaluation closure should return a score where higher is better. Metrics for
/// each trial as well as the best configuration are logged using `logger`.
pub fn grid_search<F>(
    space: &SearchSpace,
    mut eval: F,
    logger: &mut Logger,
) -> (Config, f32)
where
    F: FnMut(Config) -> f32,
{
    let mut best_score = f32::NEG_INFINITY;
    let mut best_cfg = Config::default();
    let mut trial = 0;
    for &lr in &space.learning_rate {
        for &bs in &space.batch_size {
            for &ep in &space.epochs {
                for &ga in &space.gamma {
                    for &la in &space.lam {
                        for &md in &space.max_depth {
                            for &rs in &space.rollout_steps {
                                let mut cfg = Config::default();
                                cfg.learning_rate = vec![lr];
                                cfg.batch_size = bs;
                                cfg.epochs = ep;
                                cfg.gamma = ga;
                                cfg.lam = la;
                                cfg.max_depth = md;
                                cfg.rollout_steps = rs;
                                let score = eval(cfg.clone());
                                logger.log(&AutoMlRecord {
                                    trial,
                                    learning_rate: lr,
                                    batch_size: bs,
                                    epochs: ep,
                                    gamma: ga,
                                    lam: la,
                                    max_depth: md,
                                    rollout_steps: rs,
                                    score,
                                    kind: "grid",
                                });
                                if score > best_score {
                                    best_score = score;
                                    best_cfg = cfg;
                                }
                                trial += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    logger.log(&AutoMlRecord {
        trial,
        learning_rate: best_cfg.learning_rate[0],
        batch_size: best_cfg.batch_size,
        epochs: best_cfg.epochs,
        gamma: best_cfg.gamma,
        lam: best_cfg.lam,
        max_depth: best_cfg.max_depth,
        rollout_steps: best_cfg.rollout_steps,
        score: best_score,
        kind: "best",
    });
    (best_cfg, best_score)
}

/// Perform random search over the [`SearchSpace`] for a fixed number of trials.
/// Returns the best configuration found along with its score.
pub fn random_search<F, R>(
    space: &SearchSpace,
    trials: usize,
    mut eval: F,
    rng: &mut R,
    logger: &mut Logger,
) -> (Config, f32)
where
    F: FnMut(Config) -> f32,
    R: Rng + ?Sized,
{
    let mut best_score = f32::NEG_INFINITY;
    let mut best_cfg = Config::default();
    for t in 0..trials {
        let lr = *space
            .learning_rate
            .choose(rng)
            .expect("search space must contain at least one learning rate");
        let bs = *space
            .batch_size
            .choose(rng)
            .expect("search space must contain at least one batch size");
        let ep = *space
            .epochs
            .choose(rng)
            .expect("search space must contain at least one epoch value");
        let ga = *space
            .gamma
            .choose(rng)
            .expect("search space must contain at least one gamma value");
        let la = *space
            .lam
            .choose(rng)
            .expect("search space must contain at least one lambda value");
        let md = *space
            .max_depth
            .choose(rng)
            .expect("search space must contain at least one max depth");
        let rs = *space
            .rollout_steps
            .choose(rng)
            .expect("search space must contain at least one rollout step");
        let mut cfg = Config::default();
        cfg.learning_rate = vec![lr];
        cfg.batch_size = bs;
        cfg.epochs = ep;
        cfg.gamma = ga;
        cfg.lam = la;
        cfg.max_depth = md;
        cfg.rollout_steps = rs;
        let score = eval(cfg.clone());
        logger.log(&AutoMlRecord {
            trial: t,
            learning_rate: lr,
            batch_size: bs,
            epochs: ep,
            gamma: ga,
            lam: la,
            max_depth: md,
            rollout_steps: rs,
            score,
            kind: "random",
        });
        if score > best_score {
            best_score = score;
            best_cfg = cfg;
        }
    }
    logger.log(&AutoMlRecord {
        trial: trials,
        learning_rate: best_cfg.learning_rate[0],
        batch_size: best_cfg.batch_size,
        epochs: best_cfg.epochs,
        gamma: best_cfg.gamma,
        lam: best_cfg.lam,
        max_depth: best_cfg.max_depth,
        rollout_steps: best_cfg.rollout_steps,
        score: best_score,
        kind: "best",
    });
    (best_cfg, best_score)
}
