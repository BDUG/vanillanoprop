# ONNX Export Example

## Overview

Export trained weights to the ONNX format for interoperability with other
frameworks.

## Running the Example

```bash
./run.sh train-noprop --export-onnx model.onnx
```

## Explanation

The exporter supports linear, convolution, ReLU, max pooling and batch
normalization layers and writes models targeting opset 13.

## Next Steps

See the [ONNX export section](../introduction.md#onnx-export) of the
introduction for more background on supported layers and usage.
