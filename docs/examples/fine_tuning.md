# Fine-tuning Example

## Overview

Initialise a model from the Hugging Face Hub and continue training while
optionally freezing specific layers.

## Running the Example

```bash
./run.sh train-backprop --fine-tune bert-base-uncased --freeze-layers 0,1,2
```

## Explanation

The command downloads the checkpoint, loads it and updates all layers except
the first three. Expect training logs as the remaining layers are optimised.

## Next Steps

Read more about adapting pretrained models in the
[Fine-tuning section](../introduction.md#fine-tuning) of the introduction.
