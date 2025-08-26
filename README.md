# vanillanoprop

This here is a vanilla implementation of [NoProp](https://arxiv.org/html/2503.24322v2) according to *Qinyu Li*.
It is part of a baremetal AI framework, not requering exhausting 3rd party libraries, compare *Cargo.toml*. 

Call
```
./run.sh
```
to see the continously evolving command line parameters.

For example:
```
./run.sh train-noprop cnn --moe --num-experts 4
./run.sh predict --moe --num-experts 4
```
The `--moe` flag enables mixture-of-experts layers and `--num-experts` sets
how many experts to use.
