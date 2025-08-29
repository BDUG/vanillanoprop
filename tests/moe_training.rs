use vanillanoprop::layers::{Layer, LinearT, MixtureOfExpertsT};
use vanillanoprop::math::{self, Matrix};

#[test]
fn moe_layer_updates_parameters() {
    // Build mixture with two experts
    let experts: Vec<Box<dyn Layer>> = (0..2)
        .map(|_| Box::new(LinearT::new(4, 2)) as Box<dyn Layer>)
        .collect();
    let mut moe = MixtureOfExpertsT::new(4, experts, 1);
    let x = Matrix::from_vec(1, 4, vec![0.1, 0.2, 0.3, 0.4]);
    let logits = moe.forward_train(&x);
    let (_loss, grad, _preds) = math::softmax_cross_entropy(&logits, &[1], 0);

    moe.zero_grad();
    moe.backward(&grad);
    // Capture parameters before update
    let mut params = moe.parameters();
    let before: Vec<Vec<f32>> = params
        .iter()
        .map(|p| p.w.data.data.clone())
        .collect();
    for p in params.iter_mut() {
        p.sgd_step(0.1, 0.0);
    }
    let params_after = moe.parameters();
    for (after, b) in params_after.iter().zip(before.iter()) {
        let changed = after
            .w
            .data
            .data
            .iter()
            .zip(b.iter())
            .any(|(a, b)| (*a - *b).abs() > 1e-6);
        assert!(changed, "parameter did not update");
    }
}
