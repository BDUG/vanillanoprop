# ONNX Export Example

## Overview

Export trained weights to the ONNX format for interoperability with other
frameworks.

## Running the Example
=======
Training binaries can export weights to the ONNX format by providing an
output path.

**Prerequisites:** choose an output file for the exported model.

**Training command:** (use `./run.sh`; demos are run with
`cargo run --example`)

```bash
./run.sh train-noprop --export-onnx model.onnx
```

## Explanation

The exporter supports linear, convolution, ReLU, max pooling and batch
normalization layers and writes models targeting opset 13.

## Next Steps

See the [ONNX export section](../introduction.md#onnx-export) of the
introduction for more background on supported layers and usage.
