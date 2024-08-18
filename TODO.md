# TODO

## Model

- [ ] QAT: current quantization is not compatible with TensorRT.
- [x] Hairy floors: For some reason late models like to draw horizontal lines on flat surfaces. Needs investigation.
- [x] Dynamic scene detection: current model only uses global scene detection, so when things disappear (text, markers etc), it gets confused.
- [ ] Fade-ins/Fade-outs: flow model gets very confused when there is fade-ins/fade-outs.
- [ ] LSTM architecture.

## Inference

### TensorRT

- [x] Model re-packer. Remove unused weights and unused properties from raw models.
- [ ] Support QAT. Current backend does not support DQ/QT layers.
- [ ] Remove default values from model properties.

### TVM

- [ ] Model Quantization for OpenCL.

## Core

- [ ] Implement TVM backend (LLVM + OpenCL).
- [x] Add quantization options.
- [ ] Add GPU input/output tensors.

## OBS plugin

- [ ] Implement as Effect filter.
- [x] Add properties (backend + model + quantization options).

## AviSynth plugin

- [x] Add properties (backend + model + quantization options).
