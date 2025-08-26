use std::f32::consts::PI;

pub trait LearningRateSchedule {
    fn next_lr(&self, step: usize) -> f32;
}

pub struct ConstantLr {
    lr: f32,
}

impl ConstantLr {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl LearningRateSchedule for ConstantLr {
    fn next_lr(&self, _step: usize) -> f32 {
        self.lr
    }
}

pub struct StepLr {
    base_lr: f32,
    step_size: usize,
    gamma: f32,
}

impl StepLr {
    pub fn new(base_lr: f32, step_size: usize, gamma: f32) -> Self {
        Self {
            base_lr,
            step_size,
            gamma,
        }
    }
}

impl LearningRateSchedule for StepLr {
    fn next_lr(&self, step: usize) -> f32 {
        let exp = (step / self.step_size) as f32;
        self.base_lr * self.gamma.powf(exp)
    }
}

pub struct CosineLr {
    base_lr: f32,
    max_steps: usize,
}

impl CosineLr {
    pub fn new(base_lr: f32, max_steps: usize) -> Self {
        Self { base_lr, max_steps }
    }
}

impl LearningRateSchedule for CosineLr {
    fn next_lr(&self, step: usize) -> f32 {
        let t = step.min(self.max_steps) as f32 / self.max_steps as f32;
        0.5 * self.base_lr * (1.0 + (PI * t).cos())
    }
}

pub enum LrScheduleConfig {
    Constant,
    Step { step_size: usize, gamma: f32 },
    Cosine { max_steps: usize },
}

impl Default for LrScheduleConfig {
    fn default() -> Self {
        LrScheduleConfig::Constant
    }
}
