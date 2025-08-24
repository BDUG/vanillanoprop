use crate::transformer_t::LinearT;

pub struct SGD {
    pub lr: f32,
    pub weight_decay: f32,
}

impl SGD {
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        Self { lr, weight_decay }
    }

    pub fn step(&mut self, params: &mut [&mut LinearT]) {
        for p in params.iter_mut() {
            p.sgd_step(self.lr, self.weight_decay);
        }
    }
}
