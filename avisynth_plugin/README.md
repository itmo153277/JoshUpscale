# JoshUpscale AviSynth plugin

## Usage

<!-- clip JoshUpscale(clip "clip", string "model", int "quant" = 2)  -->
<code><span style="font-size:1.12em">*clip* **JoshUpscale**(*clip* **"clip"**, *string* **"model_path"**, *int* **"quant"** = 3)</span></code>

Parameters:

- *clip* clip  
  Input clip. Must be in RGB24 format, 480x270
- *string* model_path  
  Path to the `yaml` model file
- *int* quant = 1  
  Quantization type. Accepted values:
  - 0: no quantization (float32)
  - 1: float16 quantization
  - 2: int8 (+float16) quantization

Output:
- *clip*  
  Output clip in RGB24 format, 1920x1080

### Example

```avisynth
LoadPlugin("ffms2.dll")
LoadPlugin("JoshUpscaleAvisynth.dll")

FFVideoSource("input_vcs.mkv", fpsnum=30, fpsden=1).AssumeFPS(30, 1)

ConvertToRGB24()
PointResize(480, 270)

JoshUpscale("models/model_adapt.yaml", quant=1)
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

ConvertToRGB24()
PointResize(480, 270)

JoshUpscale("models/model_adapt.yaml", quant=1)

Prefetch(16)
```

But you can do this:

```avisynth
...

ConvertToRGB24()
PointResize(480, 270)

Prefetch(16)

JoshUpscale("models/model_adapt.yaml", quant=1)
```

Alternatively, you can force the first frame to be processed separately
and limit prefetch to 16 frames. But you will have to do this for every cut.

```avisynth
...

ConvertToRGB24()
PointResize(480, 270)

JoshUpscale("models/model_adapt.yaml", quant=1)

# Force processing order
last.trim(0, -1) + last.prefetch(16, 16).trim(1, 0)
```

Note, that the performance benefits of MT are minimal because `JoshUpscale`
would remain being the heaviest workload.

## OnDevice

The filter runs on CUDA but accepts and outputs CPU frames only. Performance
might become worse if you use it with filters that are not on CPU.
