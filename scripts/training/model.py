# -*- coding: utf-8 -*-

"""Model definition."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from layers import UpscaleLayer, DenseWarpLayer


def get_generator_model(name="generator", num_blocks=20, num_filters=32,
                        input_dtypes=None, output_dtype=None):
    """
    Create generator model.

    Inputs:
    - input_image (N x H x W x 3) - input frame
    - pre_warp (N x H * 2 x W * 2 x 3) - warped previously generated frame

    Outputs:
    - (N x H * 2 x W * 2 x 3) - upscaled frame

    Parameters
    ----------
    name: str
        Model name
    num_blocks: int
        Number of residual blocks
    num_filters: int
        Number of filters
    input_dtypes : array of str or None
        Data types for input images and warped previous images
    output_dtype : str or None
        Output dtype

    Returns
    -------
    keras.Model
        Model
    """
    images = keras.Input(shape=[None, None, 3],  name="input_image",
                         dtype=input_dtypes[0] if input_dtypes else None)
    pre_warp = keras.Input(shape=[None, None, 3], name="pre_warp",
                           dtype=input_dtypes[1] if input_dtypes else None)
    inputs = layers.Concatenate()(
        [images, layers.Lambda(lambda x: tf.nn.space_to_depth(x, 2))(pre_warp)]
    )
    net = layers.Conv2D(filters=num_filters, kernel_size=3,
                        strides=1, padding="SAME", use_bias=False)(inputs)
    net = layers.BatchNormalization()(net)
    net = layers.LeakyReLU()(net)

    def res_block(x):  # pylint: disable=invalid-name
        shortcut = x
        x = layers.Conv2D(filters=num_filters, kernel_size=3,
                          strides=1, padding="SAME", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(filters=num_filters, kernel_size=3,
                          strides=1, padding="SAME", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.LeakyReLU()(x)
        return x

    for _ in range(num_blocks):
        net = res_block(net)

    net = layers.Conv2DTranspose(filters=num_filters, kernel_size=3,
                                 strides=2, padding="SAME")(net)
    net = layers.LeakyReLU()(net)
    net = layers.Conv2D(filters=3, kernel_size=3, strides=1,
                        padding="SAME")(net)
    net = layers.Activation(K.tanh)(net)
    upscaled = UpscaleLayer(dtype="float32")(images)
    output = layers.Add(dtype=output_dtype)([upscaled, net])
    model = keras.Model(inputs=[images, pre_warp],
                        outputs=output, name=name)
    return model


def get_flow_model_resnet(name="flow", num_blocks=20, num_filters=32,
                          input_dtypes=None):
    """
    Create flow model (resnet).

    Inputs:
    - input_image1 (N x H x W x 3) - input image 1
    - input_image2 (N x H x W x 3) - input image 2

    Outputs:
    - (N x H x W x 2) - warp map from image 1 to image 2

    Parameters
    ----------
    name : str
        Model name
    num_blocks : int
        Number of residual blocks
    num_filters : int
        Number of filters
    input_dtypes : array of str or str or None
        Datatypes from input images

    Returns
    -------
    keras.Model
        Model
    """
    if not input_dtypes:
        input_dtypes = [None, None]
    elif isinstance(input_dtypes, str):
        input_dtypes = [input_dtypes, input_dtypes]
    input_image1 = keras.Input(shape=[None, None, 3], name="input_image1",
                               dtype=input_dtypes[0])
    input_image2 = keras.Input(shape=[None, None, 3], name="input_image2",
                               dtype=input_dtypes[1])
    inputs = layers.Concatenate()([input_image1, input_image2])
    net = layers.Conv2D(filters=num_filters, kernel_size=3,
                        strides=1, padding="SAME", use_bias=False)(inputs)
    net = layers.BatchNormalization()(net)
    net = layers.LeakyReLU()(net)

    def res_block(x):  # pylint: disable=invalid-name
        shortcut = x
        x = layers.Conv2D(filters=num_filters, kernel_size=3,
                          strides=1, padding="SAME", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(filters=num_filters, kernel_size=3,
                          strides=1, padding="SAME", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.LeakyReLU()(x)
        return x

    for _ in range(num_blocks):
        net = res_block(net)
    net = layers.Conv2D(filters=2, kernel_size=3, strides=1,
                        padding="SAME")(net)
    net = layers.Activation(K.tanh)(net)
    output = layers.Lambda(lambda x: x * 24.0)(net)
    model = keras.Model(inputs=[input_image1, input_image2], outputs=output,
                        name=name)
    return model


def get_flow_model_autoencoder(name="flow", filters=None, input_dtypes=None):
    """
    Create flow model (autoencoder).

    Inputs:
    - input_image1 (N x H x W x 3) - input image 1
    - input_image2 (N x H x W x 3) - input image 2

    Outputs:
    - (N x H x W x 2) - warp map from image 1 to image 2

    Parameters
    ----------
    name : str
        Model name
    filters : array of int
        Filters for autoencoder
    input_dtypes : array of str or str or None
        Datatypes from input images

    Returns
    -------
    keras.Model
        Model
    """
    if not input_dtypes:
        input_dtypes = [None, None]
    elif isinstance(input_dtypes, str):
        input_dtypes = [input_dtypes, input_dtypes]
    if not filters:
        filters = [32, 64, 128, 256, 128, 64, 32]
    input_image1 = keras.Input(shape=[None, None, 3], name="input_image1",
                               dtype=input_dtypes[0])
    input_image2 = keras.Input(shape=[None, None, 3], name="input_image2",
                               dtype=input_dtypes[1])
    inputs = layers.Concatenate()([input_image1, input_image2])

    def down_block(x, filters):  # pylint: disable=invalid-name
        x = layers.Conv2D(filters=filters, kernel_size=3,
                          strides=1, padding="SAME", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(filters=filters, kernel_size=3,
                          strides=1, padding="SAME", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.MaxPool2D(pool_size=2)(x)
        return x

    def up_block(x, filters):  # pylint: disable=invalid-name
        x = layers.Conv2D(filters=filters, kernel_size=3,
                          strides=1, padding="SAME", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(filters=filters, kernel_size=3,
                          strides=1, padding="SAME", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = UpscaleLayer(dtype="float32")(x)
        return x

    block_count = len(filters) // 2
    net = inputs
    for i in range(block_count):
        net = down_block(net, filters[i])
    for i in range(block_count, block_count * 2):
        net = up_block(net, filters[i])
    if len(filters) % 2:
        net = layers.Conv2D(filters=filters[-1], kernel_size=3,
                            strides=1, padding="SAME", use_bias=False)(net)
        net = layers.BatchNormalization()(net)
        net = layers.LeakyReLU()(net)
    net = layers.Conv2D(filters=2, kernel_size=3, strides=1,
                        padding="SAME")(net)
    net = layers.Activation(K.tanh)(net)
    output = layers.Lambda(lambda x: x * 24.0)(net)
    model = keras.Model(inputs=[input_image1, input_image2], outputs=output,
                        name=name)
    return model


def get_discriminator_model(name="discriminator", crop_size=32):
    """
    Create discriminator model.

    Inputs:
    - input  (N x H * 2 x W * 2 x 27) - input

    Outputs:
    - (N x H x W x 64) - layer 1 output
    - (N x H / 2 x W / 2 x 64) - layer 2 output
    - (N x H / 4 x W / 4 x 128) - layer 3 output
    - (N x H / 8 x W / 8 x 256) - layer 4 output
    - (N x H / 8 x W / 8 x 1) - fake / real discrimination

    Parameters
    ----------
    name : str
        Model name
    crop_size : int
        Crop size

    Returns
    -------
    keras.Model
        Model
    """
    inputs = keras.Input(shape=[crop_size * 2, crop_size * 2, 27],
                         name="input")
    outputs = []
    net = layers.Conv2D(filters=64, kernel_size=3,
                        strides=1, padding="SAME")(inputs)

    def discriminator_block(x, filters):  # pylint: disable=invalid-name
        x = layers.Conv2D(filters=filters, kernel_size=4,
                          strides=2, padding="SAME", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        return x

    net = layers.LeakyReLU()(net)
    net = discriminator_block(net, 64)
    outputs.append(net)
    net = discriminator_block(net, 64)
    outputs.append(net)
    net = discriminator_block(net, 128)
    outputs.append(net)
    net = discriminator_block(net, 256)
    outputs.append(net)
    net = layers.Dense(1)(net)
    net = layers.Activation(tf.nn.sigmoid)(net)
    outputs.append(net)
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def get_full_model(generator_model, flow_model, name="full", dtype=None):
    """
    Create full model.

    Inputs:
    - cur_frame (N x H x W x 3) - frame to process
    - last_frame (N x H x W x 3) - previously processed frame
    - pre_gen (N x H * 2 x W * 2 x 3) - result of the previous processing

    Outputs:
    - (N x H * 2 x W * 2 x 3) - upscaled frame
    - (N x H * 2 x W * 2 x 3) - warped previously upscaled frame

    Parameters
    ----------
    generator_model : keras.Model
        Generator model
    flow_model : keras.Model
        Flow model
    name : str
        Model name
    dtype : str or None
        Input and output type

    Returns
    -------
    keras.Model
        Model
    """
    cur_frame = keras.Input(shape=[None, None, 3], dtype=dtype,
                            name="cur_frame")
    last_frame = keras.Input(shape=[None, None, 3], dtype=dtype,
                             name="last_frame")
    pre_gen = keras.Input(shape=[None, None, 3], dtype=dtype,
                          name="pre_gen")
    flow = flow_model([last_frame, cur_frame])
    flow = UpscaleLayer(dtype="float32")(flow)
    flow = layers.Lambda(lambda x: x * 2, dtype="float32")(flow)
    pre_warp = DenseWarpLayer(dtype="float32")([pre_gen, flow])
    output = generator_model([cur_frame, pre_warp])
    model = keras.Model(inputs=[cur_frame, last_frame, pre_gen],
                        outputs=[output, pre_warp],
                        name=name)
    return model


def get_frvsr(generator_model, flow_model, crop_size=32, learning_rate=0.0005,
              steps_per_execution=1):
    """
    Create FRVSR model.

    Inputs:
    - input (N x 10 x H x W x 3) - input frames
    - target (N x 10 x H * 2 x W * 2 x 3) - target frames

    Outputs:
    - gen_outputs (N x 10 x H * 2 x W * 2 x 3) - upscaled frames
    - target_warp (N x 10 x H * 2 x W * 2 x 3) - warped target frames

    Parameters
    ----------
    generator_model : keras.Model
        Generator
    flow_model : keras.Model
        Flow
    crop_size : int
        Image crop size
    learning_rate : float
        Learning rate
    steps_per_execution : int
        Steps per execution

    Returns
    -------
    keras.Model
        Model
    """
    inputs = keras.Input(shape=[10, crop_size, crop_size, 3],
                         name="input")
    targets = keras.Input(shape=[10, crop_size*2, crop_size*2, 3],
                          name="target")
    input_frames = tf.reshape(inputs[:, 1:, :, :, :],
                              [-1, crop_size, crop_size, 3])
    input_frames_pre = tf.reshape(inputs[:, :-1, :, :, :],
                                  [-1, crop_size, crop_size, 3])
    target_frames_pre = tf.reshape(targets[:, :-1, :, :, :],
                                   [-1, crop_size*2, crop_size*2, 3])
    flow_lr = flow_model([input_frames_pre, input_frames])
    flow = UpscaleLayer()(flow_lr) * 2
    target_warp = DenseWarpLayer()([target_frames_pre, flow])
    target_warp = tf.reshape(target_warp,
                             [-1, 9, crop_size*2, crop_size*2, 3])
    flow = tf.reshape(flow, [-1, 9, crop_size*2, crop_size*2, 2])
    last_output = generator_model([
        inputs[:, 0, :, :, :],
        tf.zeros_like(targets[:, 0, :, :, :])
    ])
    gen_outputs = [last_output]
    for frame_i in range(9):
        cur_flow = flow[:, frame_i, :, :, :]
        gen_pre_output_warp = DenseWarpLayer()(
            [last_output, cur_flow])
        last_output = generator_model([
            inputs[:, frame_i + 1, :, :, :],
            gen_pre_output_warp
        ])
        gen_outputs.append(last_output)
    gen_outputs = tf.reshape(tf.stack(gen_outputs, axis=1),
                             [-1, 10, crop_size*2, crop_size*2, 3])
    target_warp = layers.Layer(name="target_warp")(target_warp)
    gen_outputs = layers.Layer(name="gen_outputs")(gen_outputs)
    model = keras.Model(inputs=[inputs, targets],
                        outputs=[gen_outputs, target_warp])
    model.compile(
        loss=["mse", "mse"],
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        steps_per_execution=steps_per_execution
    )
    return model


def get_gan_model(generator_model, flow_model, discriminator_model,
                  crop_size=32):
    """
    Create GAN model.

    Inputs:
    - input (N x 10 x H x W x 3) - input frames
    - target (N x 10 x H * 2 x W * 2 x 3) - target frames

    Outputs:
    - (N x 19 x H * 2 x W * 2 x 3) - upscaled frames (bidirectional)
    - (N x 18 x H * 2 x W * 2 x 3) - warped target frames (bidirectional)
    - discriminator output on fake frames:
        - (N x 6 x H x W x 64) - layer 1 output
        - (N x 6 x H / 2 x W / 2 x 64) - layer 2 output
        - (N x 6 x H / 4 x W / 4 x 128) - layer 3 output
        - (N x 6 x H / 8 x W / 8 x 256) - layer 4 output
        - (N x 6 x H / 8 x W / 8 x 1) - fake / real discrimination
    - discriminator output on real frames:
        - (N x 6 x H x W x 64) - layer 1 output
        - (N x 6 x H / 2 x W / 2 x 64) - layer 2 output
        - (N x 6 x H / 4 x W / 4 x 128) - layer 3 output
        - (N x 6 x H / 8 x W / 8 x 256) - layer 4 output
        - (N x 6 x H / 8 x W / 8 x 1) - fake / real discrimination
    - (N x 18 x H * 2 x W * 2 x 3) - warped upscaled frames (bidirectional)

    Parameters
    ----------
    generator_model : keras.Model
        Generator
    flow_model : keras.Model
        Flow
    discriminator_model : keras.Model
        Discriminator
    crop_size : int
        Image crop size

    Returns
    -------
    keras.Model
        Model
    """
    orig_inputs = keras.Input(
        shape=[10, crop_size, crop_size, 3], name="input")
    orig_targets = keras.Input(
        shape=[10, crop_size*2, crop_size*2, 3], name="target")
    inputs_rev = orig_inputs[:, -2::-1, :, :, :]
    targets_rev = orig_targets[:, -2::-1, :, :, :]
    inputs = layers.Concatenate(axis=1)([orig_inputs, inputs_rev])
    targets = layers.Concatenate(axis=1)([orig_targets, targets_rev])
    input_frames = tf.reshape(inputs[:, 1:, :, :, :],
                              [-1, crop_size, crop_size, 3])
    input_frames_pre = tf.reshape(inputs[:, :-1, :, :, :],
                                  [-1, crop_size, crop_size, 3])
    target_frames_pre = tf.reshape(targets[:, :-1, :, :, :],
                                   [-1, crop_size*2, crop_size*2, 3])
    flow = flow_model([input_frames_pre, input_frames])
    flow = UpscaleLayer(dtype="float32")(flow) * 2
    target_warp = DenseWarpLayer(dtype="float32")([target_frames_pre, flow])
    target_warp = tf.reshape(target_warp,
                             [-1, 18, crop_size*2, crop_size*2, 3])
    flow = tf.reshape(flow, [-1, 18, crop_size*2, crop_size*2, 2])
    last_output = generator_model([
        inputs[:, 0, :, :, :],
        tf.zeros_like(targets[:, 0, :, :, :])
    ])
    gen_outputs = [last_output]
    warp_outputs = []
    for frame_i in range(18):
        cur_flow = flow[:, frame_i, :, :, :]
        gen_pre_output_warp = DenseWarpLayer(
            dtype="float32")([last_output, cur_flow])
        last_output = generator_model(
            [inputs[:, frame_i + 1, :, :, :], gen_pre_output_warp])
        gen_outputs.append(last_output)
        warp_outputs.append(gen_pre_output_warp)
    gen_outputs = tf.reshape(tf.stack(gen_outputs, axis=1),
                             [-1, 19, crop_size*2, crop_size*2, 3])
    warp_outputs = tf.reshape(tf.stack(warp_outputs, axis=1),
                              [-1, 18, crop_size*2, crop_size*2, 3])
    t_gen_outputs = tf.reshape(gen_outputs[:, :18, :, :, :],
                               [-1, crop_size*2, crop_size*2, 3])
    t_targets = tf.reshape(targets[:, :18, :, :, :],
                           [-1, crop_size*2, crop_size*2, 3])
    t_inputs = tf.reshape(inputs[:, :18, :, :, :],
                          [-1, crop_size, crop_size, 3])
    inputs_hi = UpscaleLayer(dtype="float32")(t_inputs)
    inputs_hi = tf.reshape(inputs_hi, [-1, 3, crop_size*2, crop_size*2, 3])
    inputs_hi = tf.transpose(inputs_hi, [0, 2, 3, 4, 1])
    inputs_hi = tf.reshape(inputs_hi, [-1, crop_size*2, crop_size*2, 9])
    t_inputs_vpre_batch = flow[:, :18:3, :, :, :]
    t_inputs_v_batch = tf.zeros_like(t_inputs_vpre_batch)
    t_inputs_vnxt_batch = flow[:, -2:-19:-3, :, :, :]
    t_vel = tf.stack([t_inputs_vpre_batch, t_inputs_v_batch,
                      t_inputs_vnxt_batch], axis=2)
    t_vel = tf.reshape(t_vel, [-1, crop_size*2, crop_size*2, 2])
    t_vel = tf.stop_gradient(t_vel)

    def get_warp(inputs):
        warp = DenseWarpLayer(dtype="float32")([inputs, t_vel])
        warp = tf.reshape(warp, [-1, 3, crop_size*2, crop_size*2, 3])
        warp = tf.transpose(warp, [0, 2, 3, 4, 1])
        warp = tf.reshape(warp, [-1, crop_size*2, crop_size*2, 9])
        work_size = int(crop_size * 2 * 0.75)
        pad_size = crop_size - work_size // 2
        warp = warp[:, pad_size:pad_size+work_size,
                    pad_size:pad_size+work_size, :]
        warp = tf.pad(warp, [[0, 0], [pad_size, pad_size],
                             [pad_size, pad_size], [0, 0]], "CONSTANT")
        before_warp = tf.reshape(inputs, [-1, 3, crop_size*2, crop_size*2, 3])
        before_warp = tf.transpose(before_warp, [0, 2, 3, 4, 1])
        before_warp = tf.reshape(before_warp,
                                 [-1, crop_size*2, crop_size*2, 9])
        warp = layers.Concatenate()([before_warp, warp, inputs_hi])
        return warp

    real_warp = get_warp(t_targets)
    real_output = discriminator_model(real_warp)
    real_output = (
        tf.reshape(real_output[0],
                   [-1, 6, crop_size, crop_size, 64]),
        tf.reshape(real_output[1],
                   [-1, 6, crop_size // 2, crop_size // 2, 64]),
        tf.reshape(real_output[2],
                   [-1, 6, crop_size // 4, crop_size // 4, 128]),
        tf.reshape(real_output[3],
                   [-1, 6, crop_size // 8, crop_size // 8, 256]),
        tf.reshape(real_output[4],
                   [-1, 6, crop_size // 8, crop_size // 8, 1]),
    )
    fake_warp = get_warp(t_gen_outputs)
    fake_output = discriminator_model(fake_warp)
    fake_output = (
        tf.reshape(fake_output[0],
                   [-1, 6, crop_size, crop_size, 64]),
        tf.reshape(fake_output[1],
                   [-1, 6, crop_size // 2, crop_size // 2, 64]),
        tf.reshape(fake_output[2],
                   [-1, 6, crop_size // 4, crop_size // 4, 128]),
        tf.reshape(fake_output[3],
                   [-1, 6, crop_size // 8, crop_size // 8, 256]),
        tf.reshape(fake_output[4],
                   [-1, 6, crop_size // 8, crop_size // 8, 1]),
    )
    model = keras.Model(inputs=[orig_inputs, orig_targets],
                        outputs=[gen_outputs, target_warp, fake_output,
                                 real_output, warp_outputs])
    return model
