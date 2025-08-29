use vanillanoprop::export::onnx::export_to_onnx;
use vanillanoprop::layers::{BatchNorm, MaxPool2d, ReLUT};
use vanillanoprop::models::Sequential;
use onnx_pb::ModelProto;
use prost::Message;

#[test]
fn export_includes_common_layers() {
    let mut model = Sequential::new();
    model.add_layer(Box::new(ReLUT::new()));
    model.add_layer(Box::new(MaxPool2d::new(2, 2)));
    model.add_layer(Box::new(BatchNorm::new(1, 1e-5, 0.9)));

    let path = std::env::temp_dir().join("onnx_test_model.onnx");
    export_to_onnx(&model, &path).expect("export failed");
    let data = std::fs::read(&path).expect("read onnx");
    let proto = ModelProto::decode(&*data).expect("decode onnx");
    let graph = proto.graph.expect("graph");
    let ops: Vec<_> = graph.node.iter().map(|n| n.op_type.as_str()).collect();
    assert_eq!(ops, vec!["Relu", "MaxPool", "BatchNormalization"]);
    let _ = std::fs::remove_file(path);
}
