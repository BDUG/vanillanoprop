use std::fs;
use std::path::Path;
use hf_hub::api::sync::ApiBuilder;

use vanillanoprop::models::TransformerEncoder;
use vanillanoprop::weights::load_transformer_from_hf;
use vanillanoprop::math::Matrix;

#[test]
fn hf_loading() {
    // Use local config and download weights from the Hugging Face Hub
    let cfg = Path::new("tests/data/tiny_bert/config.json");
    let api = ApiBuilder::new()
        .build()
        .expect("failed to build HF API");
    let repo = api.model("hf-internal-testing/tiny-random-bert".to_string());
    let weights = repo
        .get("model.safetensors")
        .expect("failed to fetch model weights");

    // Build model matching the config
    let mut enc = TransformerEncoder::new(2, 1000, 32, 4, 64, 0.0);

    // Load weights from safetensors
    load_transformer_from_hf(cfg, &weights, &mut enc)
        .expect("failed to load transformer weights");

    // Check embedding dimensions
    assert_eq!(enc.embedding.table.w.data.rows, 1000);
    assert_eq!(enc.embedding.table.w.data.cols, 32);

    // Check layer parameters
    for layer in &enc.layers {
        assert_eq!(layer.attn.wq.w.data.rows, 32);
        assert_eq!(layer.attn.wq.w.data.cols, 32);
        assert_eq!(layer.attn.wk.w.data.rows, 32);
        assert_eq!(layer.attn.wk.w.data.cols, 32);
        assert_eq!(layer.attn.wv.w.data.rows, 32);
        assert_eq!(layer.attn.wv.w.data.cols, 32);
        assert_eq!(layer.attn.wo.w.data.rows, 32);
        assert_eq!(layer.attn.wo.w.data.cols, 32);
        assert_eq!(layer.ff.w1.w.data.rows, 32);
        assert_eq!(layer.ff.w1.w.data.cols, 64);
        assert_eq!(layer.ff.w2.w.data.rows, 64);
        assert_eq!(layer.ff.w2.w.data.cols, 32);
    }

    // Prepare one-hot inputs for token sequence [1,2,3,4]
    let tokens = [1usize, 2, 3, 4];
    let mut x = Matrix::zeros(tokens.len(), 1000);
    for (i, &t) in tokens.iter().enumerate() {
        x.set(i, t, 1.0);
    }
    let out = enc.forward(x, None);

    // Load reference output
    let ref_text = fs::read_to_string("tests/data/tiny_bert/reference.json")
        .expect("failed to read reference output");
    let reference: Vec<Vec<f32>> = serde_json::from_str(&ref_text)
        .expect("failed to parse reference JSON");

    for i in 0..tokens.len() {
        for j in 0..32 {
            let diff = (out.data.get(i, j) - reference[i][j]).abs();
            assert!(diff < 1e-4, "mismatch at ({},{}) diff {}", i, j, diff);
        }
    }
}
