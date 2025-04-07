# JoshUpscale AviSynth plugin

## Usage

<!-- clip JoshUpscale(clip "clip", string "model", int "device" = 0)  -->
<code><span style="font-size:1.12em">*clip* **JoshUpscale**(*clip* **"clip"**, *string* **"model_path"**, *int* **"device"** = 0)</span></code>

Parameters:

- *clip* clip  
  Input clip. Must be in RGB32 format, 480x270
- *string* model_path  
  Path to TensorRT model file
- *int* device = 0
  Device index

Output:
- *clip*  
  Output clip in RGB32 format, 1920x1080

### Example

```avisynth
LoadPlugin("ffms2.dll")
LoadPlugin("JoshUpscaleAvisynth.dll")

FFVideoSource("input_vcs.mkv", fpsnum=30, fpsden=1).AssumeFPS(30, 1)

ConvertToRGB32()
PointResize(480, 270)

JoshUpscale("model.trt")
```

## Frame Access

> [!CAUTION]
> Avoid jumping more than 16 frames unless it is absolutely necessary.

The filter requires **sequential** frame access. When a jump is detected,
it will go 16 frames back to try to restore continuity.

Additionally, the last 16 frames are cached (not including "warm-up" frames).

## MT

MT should be avoided. MT might access frames in random order and
it can cause unnecessary backtracking and continuity errors.

**Do not do this**:

```avisynth
...

ConvertToRGB32()
PointResize(480, 270)

JoshUpscale("model.trt")

Prefetch(16)
```

But you can do this:

```avisynth
...

ConvertToRGB32()
PointResize(480, 270)

Prefetch(16)

JoshUpscale("model.trt")
```

Note, that the performance benefits of MT are minimal because `JoshUpscale`
would remain being the heaviest workload.

## OnDevice

This filter supports CUDA devices, so if your AviSynth is built with CUDA
enabled, you can do this:

```avisynth
# cpu filters
# ...

OnCPU(16)

# cuda filters
# ...

JoshUpscale("model.trt")

# other cuda filters
# ...

OnCUDA

# cpu filters
# ...

```
