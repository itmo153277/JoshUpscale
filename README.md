# JoshUpscale

Upscales and refines video (480x270 -> 1920x1080) using AI model.

Created for [Joshimuz](https://joshimuz.com) for upscaling GTA VCS streams.

[Demo](https://youtu.be/54DmF-qzqFY) from a live stream (sadly, the plugin wasn't working properly).

The model is originally based on [TecoGAN project](https://github.com/thunil/TecoGAN) by Mengyu Chu, et al.

This repository contains the following:

- avisynth plugin for video upscaling;
- OBS plugin for upscaling in real time;
- scripts and tools for model training.

## How to Build

Tested with Visual Studio 2022 (17.10.3) and g++ 12.3.0.

The project can be built using [CMake](https://cmake.org/). It is strongly recommended to use [vcpkg](https://vcpkg.io/) for build-time dependencies.

Additionally you will need NVIDIA CUDA Toolkit and NVIDIA TensorRT SDK.

### Plugin installer

For building installer, you need to install [Inno Setup](https://jrsoftware.org/isinfo.php).

Please put your models into `data` subfolder.

### Compatibility with NVIDIA Maxine Broadcast SDK

[NVIDIA Maxine Broadcast  SDK](https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-sdk/resources/) also uses TensorRT and it can lead to compatibility issues when OBS is trying to load it together with my plugin.

Therefore, it must be ensured that both use same TensorRT version.

To do this, you must use CUDA 11 and TensorRT 8.4 and set `USE_NVVFX` to `ON`.

## How to Train

All training scripts are located in [here](./scripts/training). The script supports training on CPU/GPU/TPU.

The model is trained using a two-step process:
1. FRVSR training: generator and flow networks are trained together on MSE loss.
2. GAN training: trains generator and flow networks with discriminator model using complex loss function.

The final model is exported as keras json config and weights in h5 format.

### Dataset

You can use either pairs of low resolution and high resolution images (recommended) or only high resolution images only (they will be downscaled for training). For training on TPU you should convert your dataset to tfrecords format.

### Generate model for inference

Use `export_model.py` to convert your model to ONNX format. Then you can use scripts from [here](./scripts/inference/) to optimize/tweak the models.

## Contributing

Check [CONTRIBUTING.md](CONTRIBUTING.md). If you found any issues or need help related to this project, feel free to create issues on GitHub or contact my via email.
