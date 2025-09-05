use std::fs;

use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use uuid::Uuid;

use vanillanoprop::models::{LlamaConfig, LlamaModel};
use vanillanoprop::weights::load_llama_from_hf;

fn zeros(n: usize) -> Vec<u8> {
    vec![0f32; n]
        .into_iter()
        .flat_map(|f| f.to_le_bytes())
        .collect()
}

#[test]
fn hf_llama_loading() {
    let vocab_size = 1000;
    let hidden_size = 32;
    let num_layers = 1;
    let num_heads = 4;
    let intermediate_size = 64;

    let tmp = std::env::temp_dir().join(Uuid::new_v4().to_string());
    fs::create_dir_all(&tmp).unwrap();
    let cfg_path = tmp.join("config.json");
    let weights_path = tmp.join("model.safetensors");

    let cfg = serde_json::json!({
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "intermediate_size": intermediate_size
    });
    fs::write(&cfg_path, cfg.to_string()).unwrap();

    let embed_data = zeros(vocab_size * hidden_size);
    let q_data = zeros(hidden_size * hidden_size);
    let k_data = zeros(hidden_size * hidden_size);
    let v_data = zeros(hidden_size * hidden_size);
    let o_data = zeros(hidden_size * hidden_size);
    let norm1_data = zeros(hidden_size);
    let norm2_data = zeros(hidden_size);
    let gate_data = zeros(intermediate_size * hidden_size);
    let down_data = zeros(hidden_size * intermediate_size);
    let up_data = zeros(intermediate_size * hidden_size);

    let tensors = vec![
        (
            "model.embed_tokens.weight".to_string(),
            TensorView::new(
                Dtype::F32,
                vec![vocab_size, hidden_size],
                &embed_data,
            )
            .unwrap(),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            TensorView::new(Dtype::F32, vec![hidden_size, hidden_size], &q_data).unwrap(),
        ),
        (
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            TensorView::new(Dtype::F32, vec![hidden_size, hidden_size], &k_data).unwrap(),
        ),
        (
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            TensorView::new(Dtype::F32, vec![hidden_size, hidden_size], &v_data).unwrap(),
        ),
        (
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            TensorView::new(Dtype::F32, vec![hidden_size, hidden_size], &o_data).unwrap(),
        ),
        (
            "model.layers.0.input_layernorm.weight".to_string(),
            TensorView::new(Dtype::F32, vec![hidden_size], &norm1_data).unwrap(),
        ),
        (
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            TensorView::new(Dtype::F32, vec![hidden_size], &norm2_data).unwrap(),
        ),
        (
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            TensorView::new(
                Dtype::F32,
                vec![intermediate_size, hidden_size],
                &gate_data,
            )
            .unwrap(),
        ),
        (
            "model.layers.0.mlp.down_proj.weight".to_string(),
            TensorView::new(
                Dtype::F32,
                vec![hidden_size, intermediate_size],
                &down_data,
            )
            .unwrap(),
        ),
        (
            "model.layers.0.mlp.up_proj.weight".to_string(),
            TensorView::new(
                Dtype::F32,
                vec![intermediate_size, hidden_size],
                &up_data,
            )
            .unwrap(),
        ),
    ];

    serialize_to_file(
        tensors.iter().map(|(n, v)| (n.as_str(), v)),
        None,
        &weights_path,
    )
    .unwrap();

    let cfg = LlamaConfig {
        vocab_size,
        hidden_size,
        num_heads,
        num_layers,
        intermediate_size,
    };
    let mut model = LlamaModel::new(cfg);

    load_llama_from_hf(&cfg_path, &weights_path, &mut model).unwrap();

    assert_eq!(model.embedding.table.w.shape[0], vocab_size);
    assert_eq!(model.embedding.table.w.shape[1], hidden_size);

    let layer = &model.layers[0];
    assert_eq!(layer.attn.wq.w.shape[0], hidden_size);
    assert_eq!(layer.attn.wq.w.shape[1], hidden_size);
    assert_eq!(layer.ffn.w1.w.shape[0], hidden_size);
    assert_eq!(layer.ffn.w1.w.shape[1], intermediate_size);
    assert_eq!(layer.ffn.w2.w.shape[0], intermediate_size);
    assert_eq!(layer.ffn.w2.w.shape[1], hidden_size);
    assert_eq!(layer.ffn.w3.w.shape[0], hidden_size);
    assert_eq!(layer.ffn.w3.w.shape[1], intermediate_size);
    assert_eq!(layer.norm1.weight.len(), hidden_size);
    assert_eq!(layer.norm2.weight.len(), hidden_size);
}
