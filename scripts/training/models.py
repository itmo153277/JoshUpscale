# -*- coding: utf-8 -*-

"""Module routines."""

from typing import Any, Callable, Dict, List, Union
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import schedules
from tensorflow.keras import applications
from keras_layers import SpaceToDepth, DepthToSpace, UpscaleLayer, ClipLayer, \
    PreprocessLayer, PostprocessLayer, DenseWarpLayer
from keras_models import FRVSRModelSingle, FRVSRModel, GANModel


Activation = Union[str, Dict[str, Any]]
LearningRateSchedule = Union[float, Dict[str, Any]]

ACTIVATIONS = {
    "relu": layers.ReLU,
    "lrelu": layers.LeakyReLU,
}

LR_SCHEDULES = {
    "constant": None,
    "exponential": schedules.ExponentialDecay,
    "piecewise": schedules.PiecewiseConstantDecay,
}


def get_activation(activation: Activation) -> Callable[..., layers.Layer]:
    """Get activation layer.

    Parameters
    ----------
    activation: Activation
        Activation definition

    Returns
    -------
    keras.layers.Layer
        Activation layer
    """
    if isinstance(activation, str):
        name = activation
        custom_args = {}
    elif isinstance(activation, dict):
        name = activation["name"]
        custom_args = {k: activation[k] for k in activation if k != "name"}
    else:
        raise TypeError("Unknown type")
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}")
    return lambda *args, **kwargs: ACTIVATIONS[name](
        *args, **custom_args, **kwargs)


def get_learning_rate(lr_schedule: LearningRateSchedule) -> \
        Union[float, schedules.LearningRateSchedule]:
    """Get learning rate schedule.

    Parameters
    ----------
    lr_schedule: LearningRateSchedule
        Learning rate definition

    Returns
    -------
    Union[float, schedules.LearningRateSchedule]
        Learning rate schedule
    """
    if isinstance(lr_schedule, float):
        return lr_schedule
    elif not isinstance(lr_schedule, dict):
        raise TypeError("Unknown type")
    name = lr_schedule["name"]
    if name not in LR_SCHEDULES:
        raise ValueError(f"Unknown learning rate type: {name}")
    if LR_SCHEDULES[name] is None:
        return lr_schedule["value"]
    return LR_SCHEDULES[name](
        **{k: v for k, v in lr_schedule.items() if k != "name"})


def get_scoped_name(scope: Union[str, None], name: str) -> Union[str, None]:
    """Get scoped name.

    Parameters
    ----------
    scope: Union[str, None]
        Scope
    name: str
        Layer name

    Returns
    -------
    Union[str, None]
        Scoped name
    """
    if scope is None:
        return None
    return f"{scope}_{name}"


def res_block(inp: tf.Tensor, num_filters: int,
              activation: Activation,
              name: Union[str, None] = None) -> tf.Tensor:
    """Create residual block.

    Parameters
    ----------
    inp: tf.Tensor
        Input tensor
    num_filters: int
        Number of filters
    activation: keras.layers.Layer
        Activation layer
    name: Union[str, None]
        Name

    Returns
    -------
    tf.Tensor
        Output tensor
    """
    shortcut = inp
    inp = layers.Conv2D(
        filters=num_filters,
        kernel_size=3,
        strides=1,
        padding="SAME",
        use_bias=False,
        name=get_scoped_name(name, "conv_1"),
    )(inp)
    inp = layers.BatchNormalization(
        name=get_scoped_name(name, "bn_1"),
    )(inp)
    inp = get_activation(activation)(
        name=get_scoped_name(name, "a_1"),
    )(inp)
    inp = layers.Conv2D(
        filters=num_filters,
        kernel_size=3,
        strides=1,
        padding="SAME",
        use_bias=False,
        name=get_scoped_name(name, "conv_2"),
    )(inp)
    inp = layers.BatchNormalization(
        name=get_scoped_name(name, "bn_2"),
    )(inp)
    inp = layers.Add(
        name=get_scoped_name(name, "add"),
    )([inp, shortcut])
    inp = get_activation(activation)(
        name=get_scoped_name(name, "a_2"),
    )(inp)
    return inp


def get_flow_resnet(
    num_inputs: int = 4,
    num_filters: int = 64,
    num_res_blocks: int = 10,
    activation: Activation = "relu",
    name: str = "flow"
) -> keras.Model:
    """Create flow net (resnet architecture).

    Inputs:
    - num_inputs x (N x H x W x 3) - input frames

    Outputs:
    - (N x H x W x 2) - warp map between two last frames

    Parameters
    ----------
    num_inputs: int
        Number of input frames
    num_filters: int
        Number of filters
    num_res_blocks: int
        Number of residual blocks
    activation: Activation
        Activation function
    name: str
        Model name

    Returns
    -------
    keras.Model
        Model
    """
    inputs = [
        keras.Input(
            shape=[None, None, 3],
            name=f"input_{i}",
        )
        for i in range(num_inputs)
    ]
    out = layers.Concatenate(
        name="concat",
    )(inputs)
    out = layers.Conv2D(
        filters=num_filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="conv_1",
    )(out)
    out = layers.BatchNormalization(
        name="bn_1",
    )(out)
    out = get_activation(activation)(
        name="a_1",
    )(out)
    for i in range(num_res_blocks):
        out = res_block(
            inp=out,
            num_filters=num_filters,
            activation=activation,
            name=f"block_{i + 1}",
        )
    out = layers.Conv2D(
        filters=32,
        kernel_size=1,
        padding="same",
        name="conv_2",
    )(out)
    out = DepthToSpace(
        block_size=4,
        name="depth_to_space",
    )(out)
    model = keras.Model(inputs=inputs, outputs=out, name=name)
    return model


def get_flow_autoencoder(
    num_inputs: int = 4,
    filters: Union[List[int], None] = None,
    activation: Activation = "relu",
    name: str = "flow"
) -> keras.Model:
    """Create flow net (autoencoder architecture).

    Inputs:
    - num_inputs x (N x H x W x 3) - input frames

    Outputs:
    - (N x H*4 x W*4 x 2) - warp map between two last frames

    Parameters
    ----------
    num_inputs: int
        Number of input frames
    filters: Union[List[int], None]
        Filters for autoencoder
    activation: Activation
        Activation function
    name: str
        Model name

    Returns
    -------
    keras.Model
        Model
    """
    if not filters:
        filters = [32, 64, 128, 256, 128, 64, 32]
    inputs = [
        keras.Input(
            shape=[None, None, 3],
            name=f"input_{i}",
        )
        for i in range(num_inputs)
    ]
    out = layers.Concatenate(
        name="concat",
    )(inputs)

    def down_block(out, num_filters, name):
        out = layers.Conv2D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            padding="SAME",
            use_bias=False,
            name=get_scoped_name(name, "conv_1"),
        )(out)
        out = layers.BatchNormalization(
            name=get_scoped_name(name, "bn_1"),
        )(out)
        out = get_activation(activation)(
            name=get_scoped_name(name, "a_1"),
        )(out)
        out = layers.Conv2D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            padding="SAME",
            use_bias=False,
            name=get_scoped_name(name, "conv_2"),
        )(out)
        out = layers.BatchNormalization(
            name=get_scoped_name(name, "bn_2"),
        )(out)
        out = get_activation(activation)(
            name=get_scoped_name(name, "a_2"),
        )(out)
        out = layers.MaxPool2D(
            pool_size=2,
            name=get_scoped_name(name, "max_pool"),
        )(out)
        return out

    def up_block(out, num_filters, name):
        out = layers.Conv2D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            padding="SAME",
            use_bias=False,
            name=get_scoped_name(name, "conv_1"),
        )(out)
        out = layers.BatchNormalization(
            name=get_scoped_name(name, "bn_1"),
        )(out)
        out = get_activation(activation)(
            name=get_scoped_name(name, "a_1"),
        )(out)
        out = layers.Conv2D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            padding="SAME",
            use_bias=False,
            name=get_scoped_name(name, "conv_2"),
        )(out)
        out = layers.BatchNormalization(
            name=get_scoped_name(name, "bn_2"),
        )(out)
        out = get_activation(activation)(
            name=get_scoped_name(name, "a_2"),
        )(out)
        out = UpscaleLayer(
            resize_type="bilinear",
            scale=2,
            name=get_scoped_name(name, "upscale"),
        )(out)
        return out

    block_count = len(filters) // 2
    for i in range(block_count):
        out = down_block(out, filters[i], f"block_{i + 1}")
    for i in range(block_count, block_count * 2):
        out = up_block(out, filters[i], f"block_{i + 1}")
    if len(filters) % 2:
        out = layers.Conv2D(
            filters=filters[-1],
            kernel_size=3,
            strides=1,
            padding="SAME",
            use_bias=False,
            name="conv_1",
        )(out)
        out = layers.BatchNormalization(
            name="bn_1",
        )(out)
        out = get_activation(activation)(
            name="a_1",
        )(out)
    out = layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding="SAME",
        name="conv_2",
    )(out)
    out = DepthToSpace(
        block_size=4,
        name="depth_to_space",
    )(out)
    model = keras.Model(inputs=inputs, outputs=out, name=name)
    return model


def get_generator_resnet(
    num_filters: int = 64,
    num_res_blocks: int = 24,
    activation: Activation = "relu",
    name: str = "generator"
) -> keras.Model:
    """Create generator model (resnet architecture).

    Inputs:
    - (N x H x W x 3) - Input image
    - (N x H*4 x W*4 x 3) - Warped previously generated frame

    Outputs:
    - (N x H*4 x W*4 x 3) - Upscaled image

    Parameters
    ----------
    num_filters: int
        NUmber of filters
    num_res_blocks: int
        Number of res blocks
    activation: Activation
        Activation function
    name: str
        Model name

    Returns
    -------
    keras.Model
        Model
    """
    images = keras.Input(shape=[None, None, 3],  name="input_image")
    pre_warp = keras.Input(shape=[None, None, 3], name="pre_warp")
    inputs = layers.Concatenate(
        name="concat",
    )(
        [images, SpaceToDepth(
            block_size=4,
            name="space_to_depth",
        )(pre_warp)]
    )
    out = layers.Conv2D(
        filters=num_filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="conv_1",
    )(inputs)
    out = layers.BatchNormalization(
        name="bn_1",
    )(out)
    out = get_activation(activation)(
        name="a_1",
    )(out)
    for i in range(num_res_blocks):
        out = res_block(out, num_filters, activation, f"block_{i + 1}")
    out = layers.Conv2DTranspose(
        filters=32,
        kernel_size=2,
        strides=2,
        padding="same",
        use_bias=False,
        name="conv_trans_1",
    )(out)
    out = layers.BatchNormalization(
        name="bn_2",
    )(out)
    out = get_activation(activation)(
        name="a_2",
    )(out)
    out = layers.Conv2DTranspose(
        filters=3,
        kernel_size=2,
        strides=2,
        padding="same",
        name="conv_trans_2",
    )(out)
    out = layers.Activation(
        K.tanh,
        name="a_3",
    )(out)
    upscaled = UpscaleLayer(
        scale=4,
        name="upscale",
    )(images)
    out = layers.Add(
        name="add",
    )([upscaled, out])
    out = ClipLayer(
        name="clip",
    )(out)
    model = keras.Model(inputs=[images, pre_warp], outputs=out, name=name)
    return model


def get_discriminator(
    crop_size: int,
    activation: Activation = "lrelu",
    alpha: float = 1.0,
    name: str = "discriminator"
) -> keras.Model:
    """Create discriminator model.

    Inputs:
    - (N x crop_size*4 x crop_size*4 x 27) - concatenated inputs

    Outputs:
    - (N x crop_size*4 x crop_size*4 x 64*alpha) - layer 1 output
    - (N x crop_size*2 x crop_size*2 x 64*alpha) - layer 2 output
    - (N x crop_size x crop_size x 128*alpha) - layer 3 output
    - (N x crop_size/2 x crop_size/2 x 256*alpha) - layer 4 output
    - (N x crop_size/2 x crop_size/2 x 1) - fake / real

    Parameters
    ----------
    crop_size: int
        Image size
    activation: Activation
        Activation
    alpha: float
        Weight scaling
    name: str
        Model name

    Returns
    -------
    keras.Model
        model
    """
    input_val = keras.Input(
        shape=[crop_size*4, crop_size*4, 27],
        name="input"
    )
    outputs = []
    net = input_val
    net = layers.Conv2D(
        filters=int(64 * alpha),
        kernel_size=3,
        strides=1,
        padding="SAME",
        name="conv_1"
    )(net)
    net = get_activation(activation)(
        name="a_1"
    )(net)

    def discriminator_block(out, filters, name):
        out = layers.Conv2D(
            filters=filters,
            kernel_size=4,
            strides=2,
            padding="SAME",
            use_bias=False,
            name=get_scoped_name(name, "conv")
        )(out)
        out = layers.BatchNormalization(
            name=get_scoped_name(name, "bn")
        )(out)
        out = get_activation(activation)(
            name=get_scoped_name(name, "a")
        )(out)
        return out

    net = discriminator_block(net, int(64 * alpha), "block_1")
    outputs.append(net)
    net = discriminator_block(net, int(64 * alpha), "block_2")
    outputs.append(net)
    net = discriminator_block(net, int(128 * alpha), "block_3")
    outputs.append(net)
    net = discriminator_block(net, int(256 * alpha), "block_4")
    outputs.append(net)
    net = layers.Dense(1, name="dense")(net)
    net = layers.Activation(K.sigmoid, name="a_2")(net)
    outputs.append(net)
    model = keras.Model(inputs=input_val, outputs=outputs, name=name)
    return model


def get_inference_model(
    generator_model: keras.Model,
    flow_model: keras.Model,
    skip_processing: bool = True,
    frame_height: Union[int, None] = None,
    frame_width: Union[int, None] = None,
    flow_pad_factor: Union[int, None] = None,
    name: str = "inference"
) -> keras.Model:
    """Create inference model.

    Inputs:
    - (N x H x W x 3) - current frame (uint8 or float32 if skip_processing)
    - (N x H*4 x W*4 x 3) - previously generated frame (float32)
    - num_flow_frames-1 x (N x PH x PW x 3) - previous frames (float32)

    Outputs:
    - output: (N x H*4 x W*4 x 3) - upscaled frame (uint8)
                                    omitted if skip_processing
    - output_raw: (N x H*4 x W*4 x 3) - upscaled frame (float32)
    - pre_warp: (N x H*4 x W*4 x 3) - warped previous upscaled frame
    - last_frames: num_flow_frames-1 x (N x PH x PW x 3) - new previous frames
                                                           (float32)

    Parameters
    ----------
    generator_model: keras.Model
        Generator model
    flow_model: keras.Model
        Flow model
    skip_processing: bool
        Skip pre/postprocessing
    frame_height: Union[int, None]
        Frame height
    frame_width: Union[int, None]
        Frame width
    flow_pad_factor: Union[int, None]
        Pad frame sizes for flow model

    Returns
    -------
    keras.Model
        Inference model
    """
    frame_shape = [frame_height, frame_width, 3]
    upscaled_shape = [
        frame_height * 4 if frame_height is not None else None,
        frame_width * 4 if frame_width is not None else None,
        3,
    ]
    if flow_pad_factor is not None:
        if frame_width is None or frame_height is None:
            raise ValueError("Width and height have to be defined")
        padded_width = ((frame_width + flow_pad_factor - 1) //
                        flow_pad_factor) * flow_pad_factor
        padded_height = ((frame_height + flow_pad_factor - 1) //
                         flow_pad_factor) * flow_pad_factor
    else:
        padded_width = frame_width
        padded_height = frame_height

    frame_dtype = "float32" if skip_processing else "uint8"
    cur_frame = keras.Input(
        shape=frame_shape,
        name="cur_frame",
        dtype=frame_dtype
    )
    pre_gen = keras.Input(
        shape=upscaled_shape,
        name="pre_gen",
        dtype="float32"
    )
    last_frames = [
        keras.Input(
            shape=[padded_height, padded_width, 3],
            name=f"last_frame_{i}",
            dtype="float32"
        )
        for i in range(len(flow_model.inputs) - 1)
    ]
    if skip_processing:
        cur_frame_proc = cur_frame
    else:
        cur_frame_proc = PreprocessLayer(
            name="preprocess",
        )(cur_frame)
    if padded_width == frame_width and padded_height == frame_height:
        cur_frame_pad = cur_frame_proc
    else:
        pad_height = padded_height - frame_height
        pad_width = padded_width - frame_width
        cur_frame_pad = layers.ZeroPadding2D(
            padding=(
                (pad_height // 2, pad_height - pad_height // 2),
                (pad_width // 2, pad_width - pad_width // 2),
            ),
            name="pad",
        )(cur_frame_proc)
    flow = flow_model([cur_frame_pad] + last_frames)
    if padded_width != frame_width or padded_height != frame_height:
        offset_x = ((padded_width - frame_width) // 2) * 4
        offset_y = ((padded_height - frame_height) // 2) * 4
        flow = layers.Lambda(
            lambda x: x[:, offset_y:offset_y+frame_height*4,
                        offset_x:offset_x+frame_width*4, :],
            name="unpad",
        )(flow)
    pre_warp = DenseWarpLayer(
        name="dense_warp",
    )([pre_gen, flow])
    output_raw = generator_model([cur_frame_proc, pre_warp])
    output = PostprocessLayer(
        name="postprocess",
    )(output_raw)
    pre_warp = layers.Layer(name="pre_warp", dtype="float32")(pre_warp)
    output_raw = layers.Layer(name="output_raw", dtype="float32")(output_raw)
    output = layers.Layer(name="output", dtype=frame_dtype)(output)
    outputs = {}
    if not skip_processing:
        outputs["output"] = output
    outputs["output_raw"] = output_raw
    outputs["pre_warp"] = pre_warp
    outputs["last_frames"] = [cur_frame_pad] + last_frames[:-1]
    model = keras.Model(
        inputs=[cur_frame, pre_gen] + last_frames,
        outputs=outputs,
        name=name
    )
    return model


def get_frvsr_single(
    inference_model: keras.Model,
    crop_size: int,
    learning_rate: Any = 0.0005,
    steps_per_execution: Union[int, None] = None,
    name: str = "frvsr",
):
    """Get FRVSR model (single).

    Inputs:
    - input: (N x num_flow_frames x H x W x 3) - input frames
    - target: (N x H*4 x W*4 x 3) - target frame
    - last: (N x H*4 x W*4 x 3) - last frame

    Outputs:
    - gen_output: (N x H*4 x W*4 x 3) - generated frame
    - pre_warp: (N x H*4 x W*4 x 3) - warped last frame

    Parameters
    ----------
    inference_model: keras.Model
        Inference model
    crop_size: int
        Image size
    learning_rate: Any
        Learning rate
    steps_per_execution: int
        Steps per single execution

    Returns
    -------
    keras.Model
        Model
    """
    model = FRVSRModelSingle(
        inference_model=inference_model,
        crop_size=crop_size,
        name=name
    )
    model.compile(
        learning_rate=learning_rate,
        steps_per_execution=steps_per_execution
    )
    return model


def get_frvsr(
    inference_model: keras.Model,
    flow_model: keras.Model,
    generator_model: keras.Model,
    crop_size: int,
    learning_rate: LearningRateSchedule = 0.0005,
    steps_per_execution: Union[int, None] = None,
    name: str = "frvsr",
) -> keras.Model:
    """Get FRVSR model.

    Inputs:
    - input: (N x 10 x crop_size x crop_size x 3) - input frames
    - target: (N x 10 x crop_size*4 x crop_size*4 x 3) - target frames

    Outputs:
    - gen_output: (N x 10 x crop_size*4 x crop_size*4 x 3) - generated frames
    - target_warp: (N x 10 x crop_size*4 x crop_size*4 x 3) - warped target
                                                              frames

    Parameters
    ----------
    inference_model: keras.Model
        Inference model
    flow_model: keras.Model
        Flow model
    generator_model: keras.Model
        Generator model
    crop_size: int
        Image size
    learning_rate: LearningRateSchedule
        Learning rate
    steps_per_execution: Union[None, int]
        Steps per single execution

    Returns
    -------
    keras.Model
        Model
    """
    model = FRVSRModel(
        inference_model=inference_model,
        flow_model=flow_model,
        generator_model=generator_model,
        crop_size=crop_size,
        name=name
    )
    model.compile(
        learning_rate=get_learning_rate(learning_rate),
        steps_per_execution=steps_per_execution
    )
    return model


def get_vgg(
    crop_size: int,
    out_layers: Union[None, List[str]] = None,
    name: str = "vgg"
) -> keras.Model:
    """Get VGG19 model.

    Inputs:
    - (N x crop_size*4 x crop_size*4 x 3) - input image
                                            (normalized to [-0.5, 0.5] range)

    Outputs are defined via out_layers

    Parameters
    ----------
    crop_size: int
        Image size
    out_layers: Union[None, List[str]]
        List of layers from VGG19
    name: str
        Model name

    Returns
    -------
    keras.Model
        Model
    """
    input_img = keras.Input(
        shape=[crop_size*4, crop_size*4, 3],
        name="input"
    )
    out = input_img
    out = layers.Rescaling(scale=255, offset=0.5, name="rescale")(out)
    out = applications.vgg19.preprocess_input(out)
    vgg_net = applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_tensor=out,
    )
    outputs = []
    if out_layers is None:
        out_layers = [
            "block2_conv2",
            "block3_conv4",
            "block4_conv4",
            "block5_conv4",
        ]
    for layer_name in out_layers:
        outputs.append(vgg_net.get_layer(layer_name).output)
    model = keras.Model(inputs=input_img, outputs=outputs, name=name)
    model.trainable = False
    return model


def get_gan(
    inference_model: keras.Model,
    generator_model: keras.Model,
    flow_model: keras.Model,
    discriminator_model: keras.Model,
    vgg_model: keras.Model,
    crop_size: int,
    learning_rate: LearningRateSchedule = 0.0005,
    loss_config: Union[None, Dict[str, Any]] = None,
    steps_per_execution: Union[None, int] = None,
    name: str = "gan",
) -> keras.Model:
    """Get GAN model.

    Inputs:
    - input: (N x 10 x crop_size x crop_size x 3) - input frames
    - target: (N x 10 x crop_size*4 x crop_size*4 x 3) - target frames

    Outputs:
    - gen_outputs: (N x 10 x crop_size*4 x crop_size*4 x 3) - generated frames
    - gen_warp: (N x 19 x crop_size*4 x crop_size*4 x 3) - warped generated
                                                           frames
    - target_warp: (N x 18 x crop_size*4 x crop_size*4 x 3) - warped target
                                                              frames
    - real_output: discr_layers x (N x 6 x S x S x C) - discriminator output on
                                                        real data
    - fake_output: discr_layers x (N x 6 x S x S x C) - discriminator output on
                                                        fake data
    - vgg_real_output: vgg_layers x (N x 19 x H x W x C) - vgg output on
                                                           real data
    - vgg_fake_output: vgg_layers x (N x 19 x H x W x C) - vgg output on
                                                           fake data

    Parameters
    ----------
    inference_model: keras.Model
        Inference model
    generator_model: keras.Model
        Generator model
    flow_model: keras.Model
        Flow model
    discriminator_model: keras.Model
        Discriminator model
    vgg_model: keras.Model
        VGG19 model
    crop_size: int
        Image size
    learning_rate: LearningRateSchedule
        Learning rate
    loss_config: Union[None, Dict[str, Any]]
        Loss config
    steps_per_execution: Union[NOne, int]
        Steps per execution

    Returns
    -------
    keras.Model
        Model
    """
    model = GANModel(
        inference_model=inference_model,
        generator_model=generator_model,
        flow_model=flow_model,
        discriminator_model=discriminator_model,
        vgg_model=vgg_model,
        loss_config=loss_config,
        crop_size=crop_size,
        name=name,
    )
    model.compile(
        learning_rate=get_learning_rate(learning_rate),
        steps_per_execution=steps_per_execution,
    )
    return model


MODELS = {
    "flow-resnet": get_flow_resnet,
    "flow-autoencoder": get_flow_autoencoder,
    "generator-resnet": get_generator_resnet,
    "discriminator": get_discriminator,
    "inference": get_inference_model,
    "frvsr-single": get_frvsr_single,
    "frvsr": get_frvsr,
    "vgg": get_vgg,
    "gan": get_gan,
}


def create_models(config: Dict[str, Any]) -> Dict[str, keras.Model]:
    """Create models from config."""
    models = {}

    def create_model(name: str) -> keras.Model:
        if name in models:
            return models[name]
        args = config[name]
        model_type = args["name"]
        model_args = {k: args[k] for k in args if k not in ["name", "weights"]}
        for arg, val in model_args.items():
            if isinstance(val, dict) and "model" in val:
                model_args[arg] = create_model(val["model"])
        if model_type not in MODELS:
            raise ValueError(f"Unknown model type {model_type}")
        model = MODELS[model_type](name=name, **model_args)
        if "weights" in args:
            status = model.load_weights(args["weights"])
            if status is not None:
                status.expect_partial()
                del status
        models[name] = model
        return model

    for name in config:
        create_model(name)

    return models
