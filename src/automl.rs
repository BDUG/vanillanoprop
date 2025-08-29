use crate::config::Config;
use crate::logging::Logger;
use rand::{seq::SliceRandom, Rng};
use serde::Serialize;

/// Describes the hyperparameter options to explore during search.
#[derive(Debug, Clone)]
pub struct SearchSpace {
    /// Candidate learning rates.
    pub learning_rate: Vec<f32>,
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
        }
    }
}

#[derive(Serialize)]
struct AutoMlRecord {
    trial: usize,
    learning_rate: f32,
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
    for (i, lr) in space.learning_rate.iter().copied().enumerate() {
        let mut cfg = Config::default();
        cfg.learning_rate = vec![lr];
        let score = eval(cfg.clone());
        logger.log(&AutoMlRecord {
            trial: i,
            learning_rate: lr,
            score,
            kind: "grid",
        });
        if score > best_score {
            best_score = score;
            best_cfg = cfg;
        }
    }
    logger.log(&AutoMlRecord {
        trial: space.learning_rate.len(),
        learning_rate: best_cfg.learning_rate[0],
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
        let mut cfg = Config::default();
        cfg.learning_rate = vec![lr];
        let score = eval(cfg.clone());
        logger.log(&AutoMlRecord {
            trial: t,
            learning_rate: lr,
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
        score: best_score,
        kind: "best",
    });
    (best_cfg, best_score)
}
