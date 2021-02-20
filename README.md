# JoshUpscale

Upscales and refines video (480x272 -> 960x544) using AI model. Originally created for upscaling GTA VCS stream but may be used for other things as well.

Demo:

- large model: https://youtu.be/CYsiFPwO6P0
- small model: https://youtu.be/aBkcG-yTNts

The model is originally based on [TecoGAN project](https://github.com/thunil/TecoGAN) by Mengyu Chu, et al.

This repository contains the following:

- avisynth plugin for video upscaling;
- GUI program that captures video and applies upscaling in real time;
- scripts and tools for model training and dataset generation (WIP).

## How to Build

Supported compilers: Visual Studio 2019 (16.8.4).

### 1. Install Dependencies

Before building you need to install the following dependencies:

1. TensorFlow 2.3.1
    - `%TFPath%\include` &mdash; for headers
    - `%TFPath%\lib` &mdash; for library files
2. SDL 2.0.14
    - `%SDLPath%\include` &mdash; for headers
    - `%SDLPath%\lib` &mdash; for library files
3. FFmpeg 4.3
    - `%FFPath%\include` &mdash; for headers
    - `%FFPath%\lib` &mdash; for library files
4. wxWidgets 3.1.4
    - `%wxPath%\wxwidgets.props` &mdash; for wxWidgets build configuration properties
5. AviSynth+ FilterSDK 3.6.1
    - `%AviSynthPath%\include` &mdash; for headers
    - `%AviSynthPath%\lib` &mdash; for library files

Ways to provide the paths:

1. Using global properties (recommended):

    Edit your `Microsoft.Cpp.$(Platform).user.props` and add user macros for `TFPath`, `SDLPath`, `FFPath`, `wxPath` and `AviSynthPath`

2. Using environment variables `TFPath`, `SDLPath`, `FFPath`, `wxPath` and `AviSynthPath`

3. Using project-local variables:

    Create file `src/build/deps.props.user` with the following content:

    ```xml
    <?xml version="1.0" encoding="utf-8"?>
    <Project ToolsVersion="Current" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
    <PropertyGroup Label="UserMacros">
        <wxPath></wxPath>
        <SDLPath></SDLPath>
        <FFPath></FFPath>
        <TFPath></TFPath>
        <AviSynthPath></AviSynthPath>
    </PropertyGroup>
    </Project>
    ```

    Specify your paths in the corresponding fields.

### 2. Build Solution

Open `src/JoshUpscale.sln` (you can ignore `setup.vdproj`) and build the solution.

Output binaries will be created in the following directories:

- `src\%configuration%\gui\%platform%\JoshUpscale.exe` &mdash; GUI Launcher
- `src\%configuration%\avisynth_plugin\%platform%\JoshUpscale.dll` &mdash; Avisynth plugin
- `src\%configuration%\FrameSync\%platform%\FrameSync.dll` &mdash; Avisynth plugin for synchronising two video streams

## Model Training

Refer to [this](./scripts/training/).

## Dataset

Refer to [this](./scripts/dataset/).

## Setup Package

Refer to [this](./src/setup/).
