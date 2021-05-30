## What's this

A pytorch network (WIP) to run harris corner detection

## How to use

If your device supports pytorch's abs function, use
```
model.py --abs
```
else use
```
model.py
```

## Using the makefile

`make build` runs `make docker` + compiles the network in the docker.

In order to change to `model.py --abs`, you need to manually modify the makefile (I couldn't make it via cli arg). The target `out/model.onnx` needs to require `abs` instead of `no_abs`.

Sometimes, `make build` needs to be run twice in order to compile the model.
