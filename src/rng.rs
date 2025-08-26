use rand::{rngs::StdRng, SeedableRng};
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Create a [`StdRng`] seeded from the `SEED` environment variable.
///
/// Each call uses a unique seed derived from the base seed and an
/// incrementing counter to ensure deterministic yet distinct streams.
pub fn rng_from_env() -> StdRng {
    let base = std::env::var("SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let idx = COUNTER.fetch_add(1, Ordering::SeqCst);
    StdRng::seed_from_u64(base + idx)
}
