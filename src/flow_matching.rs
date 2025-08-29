/// A simple flow model describing a time-dependent dynamical system.
///
/// The model holds a function `f(t, x)` describing the time derivative of the
/// state `x`.  The function is integrated with a basic Euler solver which is
/// sufficient for small step sizes and keeps the implementation dependency free.
/// A helper [`time_loss`] method runs the integration and evaluates a mean
/// squared error against a target state at the final time.
pub struct FlowModel<F>
where
    F: Fn(f32, &[f32]) -> Vec<f32> + Send + Sync,
{
    /// Dynamical system describing `dx/dt = f(t, x)`.
    pub f: F,
}

impl<F> FlowModel<F>
where
    F: Fn(f32, &[f32]) -> Vec<f32> + Send + Sync,
{
    /// Create a new [`FlowModel`] from the provided differential function.
    pub fn new(f: F) -> Self {
        Self { f }
    }

    /// Integrate the system from `t0` to `t1` starting at state `x0` using an
    /// explicit Euler method with `steps` iterations.
    pub fn integrate(&self, x0: &[f32], t0: f32, t1: f32, steps: usize) -> Vec<Vec<f32>> {
        let dt = (t1 - t0) / steps as f32;
        let mut x = x0.to_vec();
        let mut t = t0;
        let mut traj = vec![x.clone()];
        for _ in 0..steps {
            let dx = (self.f)(t, &x);
            for i in 0..x.len() {
                x[i] += dx[i] * dt;
            }
            t += dt;
            traj.push(x.clone());
        }
        traj
    }

    /// Compute a simple time dependent mean squared error between the state at
    /// `t1` and a target state.  The dynamics are obtained by integrating the
    /// flow from `x0`.
    pub fn time_loss(
        &self,
        x0: &[f32],
        target: &[f32],
        t0: f32,
        t1: f32,
        steps: usize,
    ) -> f32 {
        let traj = self.integrate(x0, t0, t1, steps);
        let xt = traj.last().unwrap();
        xt.iter()
            .zip(target.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f32>()
            / xt.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::FlowModel;
    use std::f32::consts::E;

    #[test]
    fn integrates_constant_derivative() {
        let flow = FlowModel::new(|_t, _x| vec![1.0]);
        let traj = flow.integrate(&[0.0], 0.0, 1.0, 10);
        let last = traj.last().unwrap()[0];
        // dx/dt = 1 -> x(t) = t
        assert!((last - 1.0).abs() < 1e-2);
    }

    #[test]
    fn computes_loss() {
        let flow = FlowModel::new(|_t, x| x.to_vec());
        // dx/dt = x, x(0)=1 -> x(1) = e
        let loss = flow.time_loss(&[1.0], &[E], 0.0, 1.0, 1000);
        assert!(loss < 1e-2);
    }
}

