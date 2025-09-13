use std::fs;

use vanillanoprop::models::TransformerEncoder;
use vanillanoprop::weights::load_transformer_from_hf;
use vanillanoprop::math::Matrix;
use vanillanoprop::fetch_hf_files;

#[test]
fn hf_loading() {
    // Download config and weights from the Hugging Face Hub
    let files = match fetch_hf_files("hf-internal-testing/tiny-random-bert", None, None) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Skipping test: {e}");
            return;
        }
    };
    let cfg = files.config;
    let weights = files.weights;

    // Build model matching the config
    let mut enc = TransformerEncoder::new(2, 1000, 32, 4, 64, 0.0);

    // Load weights from safetensors
    load_transformer_from_hf(&cfg, &weights, &mut enc)
        .expect("failed to load transformer weights");

    // Check embedding dimensions
    assert_eq!(enc.embedding.table.w.shape[0], 1000);
    assert_eq!(enc.embedding.table.w.shape[1], 32);

    // Check layer parameters
    for layer in &enc.layers {
        assert_eq!(layer.attn.wq.w.shape[0], 32);
        assert_eq!(layer.attn.wq.w.shape[1], 32);
        assert_eq!(layer.attn.wk.w.shape[0], 32);
        assert_eq!(layer.attn.wk.w.shape[1], 32);
        assert_eq!(layer.attn.wv.w.shape[0], 32);
        assert_eq!(layer.attn.wv.w.shape[1], 32);
        assert_eq!(layer.attn.wo.w.shape[0], 32);
        assert_eq!(layer.attn.wo.w.shape[1], 32);
        assert_eq!(layer.ff.w1.w.shape[0], 32);
        assert_eq!(layer.ff.w1.w.shape[1], 64);
        assert_eq!(layer.ff.w2.w.shape[0], 64);
        assert_eq!(layer.ff.w2.w.shape[1], 32);
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
            let idx = i * 32 + j;
            let diff = (out.data[idx] - reference[i][j]).abs();
            assert!(diff < 1e-4, "mismatch at ({},{}) diff {}", i, j, diff);
        }
    }
}
