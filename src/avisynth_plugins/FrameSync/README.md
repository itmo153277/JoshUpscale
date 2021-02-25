# FrameSync

Synchronises two video clips by dropping frames that exist in one clip, but do not exist the other.

## Usage

```avisynth
FrameSync(clip clip1, clip clip2, string out_file, string out_norm)
```

Parameters:

- `clip1`: first clip
- `clip2`: second clip
- `out_file`: path for output file
- `out_norm`: path for norm list file

### Example Usage

In processing script:

```avisynth
v1 = FFVideoSource(...).ConvertToRGB24()
v2 = FFVideoSource(...).ConvertToRGB24()
FrameSync(v1, v2, "sync.avsi", "norms.txt")
```

In the following scripts:

```avisynth
v1 = FFVideoSource(...)
v2 = FFVideoSource(...)
Import("sync.avsi")
Subtract(v1, v2)  # Show difference between clips
```

## How It Works

Let's say that we have two clips, one has frames `A`, `B`, `C`, `D` and the other one has `E`, `F`, `G`, `H`, `I`.

```text
Clip 1: A B C D
Clip 2: E F G H I
```

`FrameSync` will try to synchronise these clips by dropping frames from one of the clips.

For each of the following pairs of frames it will calculate `L1` norm:

- `A` - `E`
- `A` - `F`
- `B` - `E`

The pair with the least norm value will be selected. In case of `A` - `F` pair,  `E` will be dropped, in case of `B` - `E` pair, `A` will be dropped.

The process is repeated recursively until there are no frames dropped or one of the clips has no frames left.

## Output File Format

Output file is a AviSynth script that expects two variables to be available:

- `v1`: first clip
- `v2`: second clip

The script will delete the dropped frames from these clips.

## Norm List Format

List of doubles for each frame in the synchronized clips, separated by new lines.

### Example

```text
176539
232502
2775
2.17886e+06
2.18913e+06
1.61713e+06
1.61958e+06
```
