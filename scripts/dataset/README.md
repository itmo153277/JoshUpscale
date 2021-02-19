# Dataset

JoshUpscale model training scripts use TFRecord dataset.

One dataset example should contain the following:

- `input`: Input images: 10 consecutive frames in non-upscale quality encoded as images (`png` is preferred).
- `target`: Target images: 10 consecutive frames in upscaled quality, one-to-one matching input frames.

The dataset can be created with two ways:

- Get both high-quality and low-quality footage, e.g. recorded with an emulator.
- Get high-quality footage and downscale it with nearest neighbour resizing.

The second method **is not recommended**. Downscaled video is still much better than one natively rendered in low resolution.

## How to Collect Data

### 1. Get Footage

Get the footage in both low and high resolution. You can create a movie and record the gameplay in different quality modes. As for now, ppsspp does not support recording replays out-of-the-box, but it is possible with a custom build.

### 2. Remove Duplicate Frames

The recorded footage is most likely does not have exact number of frames and thus does not comply to one-to-one matching between input and target frames.

The first thing to do is to remove duplicated frames. You can find a sample script how to do it with AviSynth [here](./01_example_remove_duplicates.avs). It uses a combination of `Dup1` and `ExactDeDup` plugins to achieve it.

### 3. Synchronise Both Videos

Now the difference between number of frames is probably even larger.

We need to synchronize the footages, and we can do it by removing the frames that exist in only one of the two. You can find an example how to do it [here](./01_example_sync.avs). It uses `FrameSync` plugin, you can find its source code in this repository.

Sometimes this script is not enough and there could be some frames that do not match between two footages. You can use `clean_sync.py` script to find these frames and remove them.

### 4. Detect Scenes

To get batches of 10 consecutive frames we need to know exactly were each scene starts and ends.

You can use [`scenedetect`](https://pypi.org/project/scenedetect/) for this.

To make the video for sync detection you can use AviSynth script similar to [this](./03_example_sync_sdetect.avs).

### 5. Generate Batches

Now you can randomly pick batches of 10 frames using scene information.

You can use `./generate_batches.py` for this purpose.

### 6. Convert Frames to Images

Add the script to generate batches to [the previous script](./03_example_sync_sdetect.avs) and convert it to images for high-resolution and low-resolution footage (e.g. with `ffmpeg`).

### 7. Convert Images to TFRecord

Finally, convert frames to the TFRecord dataset.

Sample script:

```py
def get_tfrecord(inputs, targets):
    tf_record = tf.train.Example(features=tf.train.Features(feature={
        "input": tf.train.Feature(bytes_list=tf.train.BytesList(value=inputs)),
        "target": tf.train.Feature(bytes_list=tf.train.BytesList(value=targets))
    }))
    return tf_record.SerializeToString()

def create_tfrecord(path, images):
    with tf.io.TFRecordWriter(path) as wr:
        for i in range(len(images)):
            inputs = images[i]["input"]
            targets = images[i]["target"]
            wr.write(get_tfrecord(inputs, targets))
```
