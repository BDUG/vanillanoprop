# ONNX Export Example

Training binaries can export weights to the ONNX format by providing an
output path:

```bash
./run.sh train-noprop --export-onnx model.onnx
```

Only a subset of layers is supported. The resulting model targets opset 13.
