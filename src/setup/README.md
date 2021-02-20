# Setup

This project requires a plugin for Visual Studio: [Microsoft Visual Studio Installer Projects](https://marketplace.visualstudio.com/items?itemName=VisualStudioClient.MicrosoftVisualStudio2017InstallerProjects).

## Redistributables

The setup relies on Microsoft ClickToRun bootstrapper to install redistributables. That includes Visual C++ 2019 Redistributable and NVIDIA CUDA Version 11.0.

Visual C++ 2019 Redistributable should be included with Windows SDK.

NVIDIA CUDA Version 11.0 can be added using `files/redist/cuda`. You will need to either add `files/redist` to your ClickOnce package paths or copy it to ClickOnce package directory. Refer to [this](https://docs.microsoft.com/en-us/visualstudio/deployment/creating-bootstrapper-packages) for more information.

## External Files

Put all external files to `files/external`.

External files should include:

- License
- Model (`model.pb`) created by `/scripts/training/optimize.py`
- FFmpeg libraries
- TensorFlow libraries
- wxWidget libraries
- NVIDIA cuDNN libraries (they are not included in NVIDIA CUDA Toolkit)

## Patch

For x64 builds you main need to patch ClickOnce bootstrapper.

ClickOnce bootstrapper uses x86 executable even for x64 builds. This will prevent it to correctly check whether NVIDIA CUDA Toolkit is installed because it will fail to read x64 registry (by default for x86 programs it will be redirected to syswow6432).

The patch included in the repository can be used for the bootstrapper with this SHA-256 hash: `6A306F8FC3C184BC03C4C20643E147E855DA48F89BAACFEE7C87628154E66A39`.
