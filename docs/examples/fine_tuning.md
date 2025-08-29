# Fine-tuning Example

Initialise a model from the Hugging Face Hub and continue training while
optionally freezing specific layers:

```bash
./run.sh train-backprop --fine-tune bert-base-uncased --freeze-layers 0,1,2
```

The command downloads the checkpoint, loads it and updates all layers
except the first three.
