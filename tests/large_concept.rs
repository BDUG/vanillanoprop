use std::fs;

use vanillanoprop::models::LargeConceptModel;
use vanillanoprop::weights::{save_lcm, load_lcm};

#[test]
fn forward_produces_logits() {
    let model = LargeConceptModel::new(28 * 28, 16, 10);
    let img = vec![0u8; 28 * 28];
    let (_h, logits) = model.forward(&img);
    assert_eq!(logits.len(), 10);
}

#[test]
fn save_load_roundtrip() {
    let model = LargeConceptModel::new(28 * 28, 16, 10);
    let path = "test_lcm.json";
    save_lcm(path, &model).unwrap();
    let loaded = load_lcm(path, 28 * 28, 16, 10).unwrap();
    fs::remove_file(path).ok();
    assert_eq!(loaded.b1.len(), model.b1.len());
}
