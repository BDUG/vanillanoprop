use std::any::Any;
use std::error::Error;
use std::path::Path;

use crate::layers::{BatchNorm, Conv2d, LinearT, MaxPool2d, ReLUT};
use crate::models::Sequential;

use onnx_pb::{
    attribute_proto::AttributeType,
    tensor_proto::DataType,
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    ValueInfoProto,
};
use prost::Message;

/// Export a [`Sequential`] model to an ONNX file at the given path.
///
/// The current implementation supports a minimal subset of layers and maps
/// [`LinearT`] layers to `Gemm` nodes and [`Conv2d`] layers to `Conv` nodes.
/// Additional layers fall back to identity mappings.
pub fn export_to_onnx(model: &Sequential, path: &Path) -> Result<(), Box<dyn Error>> {
    let mut graph = GraphProto {
        name: "vanillanoprop".into(),
        node: Vec::new(),
        initializer: Vec::new(),
        input: vec![ValueInfoProto {
            name: "input".into(),
            ..Default::default()
        }],
        output: Vec::new(),
        ..Default::default()
    };

    let mut prev_out = "input".to_string();

    for (idx, layer) in model.layers.iter().enumerate() {
        let out_name = format!("x{}", idx);
        let any = &**layer as &dyn Any;
        if let Some(linear) = any.downcast_ref::<LinearT>() {
            add_linear(linear, &prev_out, &out_name, &mut graph);
        } else if let Some(conv) = any.downcast_ref::<Conv2d>() {
            add_conv(conv, &prev_out, &out_name, &mut graph);
        } else if any.is::<ReLUT>() {
            add_relu(&prev_out, &out_name, &mut graph);
        } else if let Some(pool) = any.downcast_ref::<MaxPool2d>() {
            add_max_pool(pool, &prev_out, &out_name, &mut graph);
        } else if let Some(bn) = any.downcast_ref::<BatchNorm>() {
            add_batch_norm(bn, &prev_out, &out_name, &mut graph);
        } else {
            // Unsupported layer – insert identity node
            graph.node.push(NodeProto {
                op_type: "Identity".into(),
                input: vec![prev_out.clone()],
                output: vec![out_name.clone()],
                ..Default::default()
            });
        }
        prev_out = out_name;
    }

    graph.output.push(ValueInfoProto {
        name: prev_out.clone(),
        ..Default::default()
    });

    let model_proto = ModelProto {
        ir_version: 8,
        graph: Some(graph),
        opset_import: vec![OperatorSetIdProto {
            version: 13,
            domain: String::new(),
            ..Default::default()
        }],
        ..Default::default()
    };

    let mut buf = Vec::new();
    model_proto.encode(&mut buf)?;
    std::fs::write(path, buf)?;
    Ok(())
}

fn add_linear(layer: &LinearT, input: &str, output: &str, graph: &mut GraphProto) {
    let weight_name = format!("{}__w", output);
    let weight = TensorProto {
        name: weight_name.clone(),
        data_type: DataType::Float as i32,
        dims: vec![layer.w.shape[0] as i64, layer.w.shape[1] as i64],
        float_data: layer.w.data.clone(),
        ..Default::default()
    };
    graph.initializer.push(weight);
    let node = NodeProto {
        op_type: "Gemm".into(),
        input: vec![input.into(), weight_name, String::new()],
        output: vec![output.into()],
        ..Default::default()
    };
    graph.node.push(node);
}

fn add_conv(layer: &Conv2d, input: &str, output: &str, graph: &mut GraphProto) {
    let weight_name = format!("{}__w", output);
    let weight = TensorProto {
        name: weight_name.clone(),
        data_type: DataType::Float as i32,
        dims: vec![
            layer.out_channels() as i64,
            layer.in_channels() as i64,
            layer.kernel_size() as i64,
            layer.kernel_size() as i64,
        ],
        float_data: layer.w.w.data.clone(),
        ..Default::default()
    };
    graph.initializer.push(weight);
    let mut node = NodeProto {
        op_type: "Conv".into(),
        input: vec![input.into(), weight_name],
        output: vec![output.into()],
        attribute: Vec::new(),
        ..Default::default()
    };
    node.attribute.push(AttributeProto {
        name: "strides".into(),
        r#type: AttributeType::Ints as i32,
        ints: vec![layer.stride() as i64, layer.stride() as i64],
        ..Default::default()
    });
    node.attribute.push(AttributeProto {
        name: "pads".into(),
        r#type: AttributeType::Ints as i32,
        ints: vec![
            layer.padding() as i64,
            layer.padding() as i64,
            layer.padding() as i64,
            layer.padding() as i64,
        ],
        ..Default::default()
    });
    graph.node.push(node);
}

fn add_relu(input: &str, output: &str, graph: &mut GraphProto) {
    graph.node.push(NodeProto {
        op_type: "Relu".into(),
        input: vec![input.into()],
        output: vec![output.into()],
        ..Default::default()
    });
}

fn add_max_pool(layer: &MaxPool2d, input: &str, output: &str, graph: &mut GraphProto) {
    let mut node = NodeProto {
        op_type: "MaxPool".into(),
        input: vec![input.into()],
        output: vec![output.into()],
        attribute: Vec::new(),
        ..Default::default()
    };
    node.attribute.push(AttributeProto {
        name: "kernel_shape".into(),
        r#type: AttributeType::Ints as i32,
        ints: vec![layer.kernel() as i64, layer.kernel() as i64],
        ..Default::default()
    });
    node.attribute.push(AttributeProto {
        name: "strides".into(),
        r#type: AttributeType::Ints as i32,
        ints: vec![layer.stride() as i64, layer.stride() as i64],
        ..Default::default()
    });
    graph.node.push(node);
}

fn add_batch_norm(layer: &BatchNorm, input: &str, output: &str, graph: &mut GraphProto) {
    let scale_name = format!("{}__scale", output);
    let scale = TensorProto {
        name: scale_name.clone(),
        data_type: DataType::Float as i32,
        dims: vec![layer.gamma.w.len() as i64],
        float_data: layer.gamma.w.clone(),
        ..Default::default()
    };
    graph.initializer.push(scale);

    let bias_name = format!("{}__bias", output);
    let bias = TensorProto {
        name: bias_name.clone(),
        data_type: DataType::Float as i32,
        dims: vec![layer.beta.w.len() as i64],
        float_data: layer.beta.w.clone(),
        ..Default::default()
    };
    graph.initializer.push(bias);

    let mean_name = format!("{}__mean", output);
    let mean = TensorProto {
        name: mean_name.clone(),
        data_type: DataType::Float as i32,
        dims: vec![layer.running_mean().len() as i64],
        float_data: layer.running_mean().to_vec(),
        ..Default::default()
    };
    graph.initializer.push(mean);

    let var_name = format!("{}__var", output);
    let var = TensorProto {
        name: var_name.clone(),
        data_type: DataType::Float as i32,
        dims: vec![layer.running_var().len() as i64],
        float_data: layer.running_var().to_vec(),
        ..Default::default()
    };
    graph.initializer.push(var);

    let mut node = NodeProto {
        op_type: "BatchNormalization".into(),
        input: vec![
            input.into(),
            scale_name,
            bias_name,
            mean_name,
            var_name,
        ],
        output: vec![output.into()],
        attribute: Vec::new(),
        ..Default::default()
    };
    node.attribute.push(AttributeProto {
        name: "epsilon".into(),
        r#type: AttributeType::Float as i32,
        f: layer.eps(),
        ..Default::default()
    });
    graph.node.push(node);
}
