# ONNX Export Example

Training binaries can export weights to the ONNX format by providing an
output path:

```bash
./run.sh train-noprop --export-onnx model.onnx
```

The exporter supports linear, convolution, ReLU, max pooling and batch
normalization layers. The resulting model targets opset 13.
