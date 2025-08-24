use crate::layers::LinearT;

pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Adam {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self { lr, beta1, beta2, eps, weight_decay }
    }

    pub fn step(&mut self, params: &mut [&mut LinearT]) {
        for p in params.iter_mut() {
            p.adam_step(self.lr, self.beta1, self.beta2, self.eps, self.weight_decay);
        }
    }
}
