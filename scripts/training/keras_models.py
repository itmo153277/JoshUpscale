# -*- coding: utf-8 -*-

"""Custom models."""

import itertools
from typing import Any, Dict, Union
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_layers import DenseWarpLayer, UpscaleLayer
from keras_metrics import CounterMetric, ExponentialMovingAvg


class JoshUpscaleModel(keras.Model):
    """JoshUpscale model."""

    # pylint: disable=invalid-name

    def __init__(self, inference_model: keras.Model, **kwargs) -> None:
        """Create JoshUpscaleModel.

        Parameters
        ----------
        inference_model: keras.Model
            Inference model
        """
        super().__init__(**kwargs)
        self.inference_model = inference_model

    def predict_step(self, data: Any) -> Dict[str, tf.Tensor]:
        """Prediction step.

        Inputs:
        - (N x 10 x H x W x 3) - Input sequence
        Outputs:
        - gen_outputs (N x 18 x H*4 x W*4 x 3) - Generated outputs
        - pre_warp (N x 17 x H*4 x W*4 x 3) - Warped outputs

        Parameters
        ----------
        data: Any
            Data for prediction (keras format)

        Returns
        -------
        Dict[str, tf.Tensor]
            Predicted values
        """
        x, _, _ = keras.utils.unpack_x_y_sample_weight(data)
        inputs = x["input"]
        targets = x["target"]
        shape = tf.shape(inputs)
        batch_size = shape[0]
        height = shape[2]
        width = shape[3]
        last_frames = [tf.zeros((batch_size, height, width, 3))] * \
            (len(self.inference_model.inputs) - 2)
        last_output = tf.zeros((batch_size, height*4, width*4, 3))
        gen_outputs = []
        pre_warps = []
        for i in itertools.chain(range(10), range(8, 0, -1)):
            cur_frame = inputs[:, i, :, :, :]
            outputs = self.inference_model(
                [cur_frame, last_output] + last_frames,
                training=False
            )
            last_output = outputs["output_raw"]
            gen_outputs.append(last_output)
            if i > 0:
                pre_warps.append(outputs["pre_warp"])
            last_frames = outputs["last_frames"]
        gen_outputs = tf.stack(gen_outputs, axis=1)
        pre_warps = tf.stack(pre_warps[1:], axis=1)
        t_inputs = tf.reshape(inputs, [-1, height, width, 3])
        t_inputs = UpscaleLayer(
            scale=4,
            resize_type="nearest",
            dtype="float32"
        )(t_inputs)
        t_inputs = tf.reshape(t_inputs, [-1, 10, height * 4, width * 4, 3])
        t_inputs_r = t_inputs[:, 8:0:-1]
        t_inputs = tf.concat([t_inputs, t_inputs_r], axis=1)
        t_targets = targets
        t_targets_r = t_targets[:, 8:0:-1]
        t_targets = tf.concat([t_targets, t_targets_r], axis=1)
        pre_warps = tf.concat(
            [t_inputs[:, 2:], pre_warps, t_targets[:, 2:]], axis=3)
        gen_outputs = tf.concat([t_inputs, gen_outputs, t_targets], axis=3)

        return {"gen_output": gen_outputs, "pre_warp": pre_warps}


class FRVSRModelSingle(keras.Model):
    """FRVSR model (single)."""

    # pylint: disable=invalid-name

    def __init__(self, inference_model: keras.Model, crop_size: int,
                 **kwargs):
        """Create FRVSRModelSingle.

        Parameters
        ----------
        inference_model: keras.Model
            Inference model
        crop_size: int
            Image size
        """
        super().__init__(
            **kwargs,
            **FRVSRModelSingle._build_model_args(
                inference_model=inference_model,
                crop_size=crop_size,
            )
        )
        self.gen_outputs_loss_tr = keras.metrics.Mean(name="gen_outputs_loss")
        self.target_warp_loss_tr = keras.metrics.Mean(name="target_warp_loss")
        self.loss_tr = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        """Get metrics."""
        return [
            self.loss_tr,
            self.gen_outputs_loss_tr,
            self.target_warp_loss_tr,
        ]

    def compile(self, learning_rate: Any = 0.0005, **kwargs) -> None:
        """Compile model.

        Parameters
        ----------
        learning_rate: Any
            Learning rate
        """
        super().compile(
            **kwargs,
            loss=None,
            optimizer=keras.optimizers.Adam(
                name="optimizer",
                learning_rate=learning_rate
            ),
        )

    def compute_loss(self, x, y, y_pred, sample_weight):
        """Compute loss."""
        assert y is None
        assert sample_weight is None
        inputs = x["input"]
        target = x["target"]
        gen_output = y_pred["gen_output"]
        pre_warp = y_pred["pre_warp"]

        # assuming that in distributed training batches are equal
        # across replicas
        batch_size = tf.shape(inputs)[0]
        num_replicas = tf.distribute.get_strategy().num_replicas_in_sync

        gen_outputs_loss = tf.math.squared_difference(gen_output, target)
        gen_outputs_loss = tf.math.reduce_sum(gen_outputs_loss, axis=-1)
        gen_outputs_loss = tf.math.reduce_mean(gen_outputs_loss)
        target_warp_loss = tf.math.squared_difference(pre_warp, target)
        target_warp_loss = tf.math.reduce_sum(target_warp_loss, axis=-1)
        target_warp_loss = tf.math.reduce_mean(target_warp_loss)
        loss = tf.add_n([gen_outputs_loss, target_warp_loss])

        self.gen_outputs_loss_tr.update_state(
            gen_outputs_loss, sample_weight=batch_size)
        self.target_warp_loss_tr.update_state(
            target_warp_loss, sample_weight=batch_size)
        self.loss_tr.update_state(loss, sample_weight=batch_size)

        # Average loss across all replicas is summed
        return tf.add_n(self.losses + [loss]) / num_replicas

    @staticmethod
    def _build_model_args(
        inference_model: keras.Model,
        crop_size: int,
    ) -> Dict[str, Any]:
        """Build model arguments."""
        num_frames = len(inference_model.inputs) - 1
        inputs = keras.Input(
            shape=[num_frames, crop_size, crop_size, 3],
            name="input",
            dtype="float32"
        )
        target = keras.Input(
            shape=[crop_size*4, crop_size*4, 3],
            name="target",
            dtype="float32"
        )
        last = keras.Input(
            shape=[crop_size*4, crop_size*4, 3],
            name="last",
            dtype="float32"
        )
        input_frames = [inputs[:, i] for i in range(num_frames)]
        output = inference_model([input_frames[-1], last] + input_frames[:-1])
        return {
            "inputs": [inputs, target, last],
            "outputs": {
                "gen_output": output["output_raw"],
                "pre_warp": output["pre_warp"],
            }
        }


class FRVSRModel(JoshUpscaleModel):
    """FRVSR model."""

    # pylint: disable=invalid-name

    def __init__(self, generator_model: keras.Model, flow_model: keras.Model,
                 crop_size: int, **kwargs) -> None:
        """Create FRVSRModel.

        Parameters
        ----------
        generator_model: keras.Model
            Generator model
        flow_model: keras.Model
            Flow model
        crop_size: int
            Image size
        inference_model: keras.Model
            Inference model (for JoshUpscaleModel)
        """
        super().__init__(**kwargs,
                         **FRVSRModel._build_model_args(
                             generator_model=generator_model,
                             flow_model=flow_model,
                             crop_size=crop_size
                         ))
        self.gen_outputs_loss_tr = keras.metrics.Mean(name="gen_outputs_loss")
        self.target_warp_loss_tr = keras.metrics.Mean(name="target_warp_loss")
        self.loss_tr = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        """Get metrics."""
        return [
            self.loss_tr,
            self.gen_outputs_loss_tr,
            self.target_warp_loss_tr,
        ]

    def compile(self, learning_rate: Any = 0.0005, **kwargs) -> None:
        """Compile model.

        Parameters
        ----------
        learning_rate: Any
            Learning rate
        """
        super().compile(
            **kwargs,
            loss=None,
            optimizer=keras.optimizers.Adam(
                name="optimizer",
                learning_rate=learning_rate
            ),
        )

    def compute_loss(self, x, y, y_pred, sample_weight):
        """Compute loss."""
        assert y is None
        assert sample_weight is None
        inputs = x["input"]
        targets = x["target"]
        gen_outputs = y_pred["gen_outputs"]
        target_warp = y_pred["target_warp"]

        # assuming that in distributed training batches are equal
        # across replicas
        batch_size = tf.shape(inputs)[0]
        num_replicas = tf.distribute.get_strategy().num_replicas_in_sync

        gen_outputs_loss = tf.math.squared_difference(gen_outputs, targets)
        gen_outputs_loss = tf.math.reduce_sum(gen_outputs_loss, axis=-1)
        gen_outputs_loss = tf.math.reduce_mean(gen_outputs_loss)
        target_warp_loss = tf.math.squared_difference(
            target_warp, targets[:, 1:])
        target_warp_loss = tf.math.reduce_sum(target_warp_loss, axis=-1)
        target_warp_loss = tf.math.reduce_mean(target_warp_loss)
        loss = tf.add_n([gen_outputs_loss, target_warp_loss])

        self.gen_outputs_loss_tr.update_state(
            gen_outputs_loss, sample_weight=batch_size * 10)
        self.target_warp_loss_tr.update_state(
            target_warp_loss, sample_weight=batch_size * 9)
        self.loss_tr.update_state(loss, sample_weight=batch_size)

        # Average loss across all replicas is summed
        return tf.add_n(self.losses + [loss]) / num_replicas

    def register_optimizer_variables(self):
        """Register optimizer variables."""
        with self.distribute_strategy.scope():
            self.optimizer.build(self.trainable_variables)

    @staticmethod
    def _build_model_args(
        generator_model: keras.Model,
        flow_model: keras.Model,
        crop_size: int,
    ) -> Dict[str, Any]:
        """Build model arguments."""
        inputs = keras.Input(shape=[10, crop_size, crop_size, 3],
                             name="input", dtype="float32")
        targets = keras.Input(shape=[10, crop_size*4, crop_size*4, 3],
                              name="target", dtype="float32")

        class InternalLayer(layers.Layer):
            """Internal layer."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.generator_model = generator_model
                self.flow_model = flow_model

            def call(self, inputs, targets):
                """Execute training inference."""
                input_shape = tf.shape(inputs[:, 0, :, :, :])
                output_shape = tf.shape(targets[:, 0, :, :, :])
                input_frames = tf.reshape(inputs[:, 1:, :, :, :],
                                          [-1, crop_size, crop_size, 3])
                input_frames_pre = tf.reshape(inputs[:, :-1, :, :, :],
                                              [-1, crop_size, crop_size, 3])
                num_rand_frames = len(flow_model.inputs) - 2
                if num_rand_frames > 0:
                    # Random last frames for flow generation
                    rand_frames = tf.reshape(
                        tf.stack([
                            tf.random.uniform(
                                input_shape,
                                minval=-0.5,
                                maxval=0.5,
                                dtype=inputs.dtype
                            )
                            for _ in range(num_rand_frames)
                        ], axis=1),
                        [-1, num_rand_frames, crop_size, crop_size, 3]
                    )
                flow_last_frames = [
                    tf.reshape(
                        tf.concat([
                            rand_frames[:, -(i + 1):, :, :, :],
                            inputs[:, :-(i + 2), :, :, :]
                        ], axis=1),
                        [-1, crop_size, crop_size, 3]
                    )
                    for i in range(num_rand_frames)
                ]
                target_frames_pre = tf.reshape(
                    targets[:, :-1, :, :, :],
                    [-1, crop_size*4, crop_size*4, 3]
                )
                flow = flow_model(
                    [input_frames, input_frames_pre] + flow_last_frames)
                target_warp = DenseWarpLayer()([target_frames_pre, flow])
                target_warp = tf.reshape(
                    target_warp, [-1, 9, crop_size*4, crop_size*4, 3])
                flow = tf.reshape(flow, [-1, 9, crop_size*4, crop_size*4, 2])
                # First frame uses random pre_warp
                last_output = generator_model([
                    inputs[:, 0, :, :, :],
                    tf.random.uniform(output_shape, minval=-0.5,
                                      maxval=0.5, dtype=inputs.dtype)
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
                gen_outputs = tf.reshape(
                    tf.stack(gen_outputs, axis=1),
                    [-1, 10, crop_size*4, crop_size*4, 3]
                )
                return [target_warp, gen_outputs]
        [target_warp, gen_outputs] = InternalLayer(
            name="internal")(inputs, targets)
        target_warp = layers.Identity(name="target_warp",
                                      dtype="float32")(target_warp)
        gen_outputs = layers.Identity(
            name="gen_outputs", dtype="float32")(gen_outputs)
        return {
            "inputs": [inputs, targets],
            "outputs": {
                "gen_outputs": gen_outputs,
                "target_warp": target_warp
            }
        }


class GANModel(JoshUpscaleModel):
    """GAN model."""

    # pylint: disable=invalid-name
    # pylint: disable=invalid-unary-operand-type

    def __init__(
        self,
        generator_model: keras.Model,
        flow_model: keras.Model,
        discriminator_model: keras.Model,
        vgg_model: keras.Model,
        crop_size: int,
        loss_config: Union[None, Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Create GANModel.

        Parameters
        ----------
        generator_model: keras.Model
            Generator model
        flow_model: keras.Model
            Flow model
        discriminator_model: keras.Model
            Discriminator model
        vgg: keras.Model
            VGG19 model
        crop_size: int
            Image size
        loss_config: Union[None, Dict[str, Any]]
            Loss configuration
        """
        super().__init__(**kwargs,
                         **GANModel._build_model_args(
                             generator_model=generator_model,
                             flow_model=flow_model,
                             discriminator_model=discriminator_model,
                             vgg_model=vgg_model,
                             crop_size=crop_size,
                         ))
        self.loss_config = GANModel._get_loss_config(loss_config)
        self.generator_layer = generator_model.name
        self.optimizer_gen = None
        self.optimizer_discr = None
        self.content_loss_tr = keras.metrics.Mean(name="content_loss")
        self.warp_loss_tr = keras.metrics.Mean(name="warp_loss")
        self.vgg_loss_tr = keras.metrics.Mean(name="vgg_loss")
        self.pp_loss_tr = keras.metrics.Mean(name="pp_loss")
        self.adv_loss_tr = keras.metrics.Mean(name="adv_loss")
        self.discr_layer_loss_tr = keras.metrics.Mean(name="discr_layer_loss")
        self.discr_real_loss_tr = keras.metrics.Mean(name="discr_real_loss")
        self.discr_fake_loss_tr = keras.metrics.Mean(name="discr_fake_loss")
        self.discr_real_acc_tr = keras.metrics.BinaryAccuracy(
            name="discr_real_acc")
        self.discr_fake_acc_tr = keras.metrics.BinaryAccuracy(
            name="discr_fake_acc")
        self.discr_steps_tr = CounterMetric(name="discr_steps")
        self.t_balance1_avg = ExponentialMovingAvg(decay=0.99)
        self.t_balance2_avg = ExponentialMovingAvg(decay=0.99)

    @property
    def metrics(self):
        """Get metrics."""
        return [
            self.content_loss_tr,
            self.warp_loss_tr,
            self.vgg_loss_tr,
            self.pp_loss_tr,
            self.adv_loss_tr,
            self.discr_layer_loss_tr,
            self.discr_real_loss_tr,
            self.discr_fake_loss_tr,
            self.discr_real_acc_tr,
            self.discr_fake_acc_tr,
            self.discr_steps_tr,
        ]

    def test_step(self, data):
        """Test step."""
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        y_pred = self(x, training=False)
        self.compute_loss(x, y, y_pred, sample_weight=sample_weight)

        result = self.compute_metrics(
            x, y, y_pred, sample_weight=sample_weight)

        # Remove discr_steps from validation metrics
        del result["discr_steps"]
        del result["t_balance1"]
        del result["t_balance2"]
        return result

    def compile(self, learning_rate: Any = 0.0005, **kwargs) -> None:
        """Compile model.

        Parameters
        ----------
        learning_rate: Any
            Learning rate
        """
        super().compile(
            **kwargs,
            loss=None,
            optimizer=keras.optimizers.Optimizer(
                name="optimizer", learning_rate=learning_rate)
        )
        self.optimizer_gen = keras.optimizers.Adam(
            name="optimizer_gen",
            learning_rate=learning_rate)
        self.optimizer_discr = keras.optimizers.Adam(
            name="optimizer_discr",
            learning_rate=learning_rate)

    def compute_loss(self, x, y, y_pred, sample_weight):
        """Compute loss."""
        assert y is None
        assert sample_weight is None
        inputs = x["input"]
        targets = x["target"]
        targets_rev = targets[:, -2::-1, :, :, :]
        targets_d = tf.concat([targets, targets_rev], axis=1)
        gen_outputs = y_pred["gen_outputs"]
        target_warp = y_pred["target_warp"]
        fake_output = y_pred["fake_output"]
        real_output = y_pred["real_output"]
        vgg_fake_output = y_pred["vgg_fake_output"]
        vgg_real_output = y_pred["vgg_real_output"]

        # assuming that in distributed training batches are equal
        # across replicas
        batch_size = tf.shape(inputs)[0]
        num_replicas = tf.distribute.get_strategy().num_replicas_in_sync

        gen_loss = []
        discr_loss = []

        content_loss = tf.math.squared_difference(gen_outputs, targets_d)
        content_loss = tf.math.reduce_sum(content_loss, axis=-1)
        content_loss = tf.math.reduce_mean(content_loss)
        if self.loss_config["content_loss"] > 0:
            gen_loss.append(self.loss_config["content_loss"] * content_loss)

        warp_loss = tf.math.squared_difference(target_warp, targets_d[:, 1:])
        warp_loss = tf.math.reduce_sum(warp_loss, axis=-1)
        warp_loss = tf.math.reduce_mean(warp_loss)
        if self.loss_config["warp_loss"] > 0:
            gen_loss.append(self.loss_config["warp_loss"] * warp_loss)

        gen_out_first = gen_outputs[:, :9, :, :, :]
        gen_out_last_rev = gen_outputs[:, -1:-10:-1, :, :, :]
        pp_loss = tf.abs(gen_out_first - gen_out_last_rev)
        pp_loss = tf.math.reduce_mean(pp_loss)
        if self.loss_config["pp_loss"] > 0:
            gen_loss.append(self.loss_config["pp_loss"] * pp_loss)

        if self.loss_config["t_balance2_threshold"] is not None:
            # Stop generator training if fake accuracy too low
            t_balance2_cond = tf.sign(
                self.t_balance2_avg.result() -
                self.loss_config["t_balance2_threshold"]
            ) / 2 + 0.5
            if self.loss_config["t_balance1_threshold"] is not None:
                # Force generator training if discriminator training
                # has stopped
                t_balance2_cond = tf.math.maximum(t_balance2_cond, tf.sign(
                    self.t_balance1_avg.result() -
                    self.loss_config["t_balance1_threshold"]
                ) / 2 + 0.5)
        else:
            t_balance2_cond = 1

        def crossentropy_loss(x):
            """Compute cross entropy loss."""
            with tf.name_scope("crossentropy_loss"):
                zeros = tf.zeros_like(x)
                cond = x >= zeros
                relu_logits = tf.where(cond, x, zeros)
                neg_abs_logits = tf.where(cond, -x, x)
                return relu_logits + tf.math.log1p(tf.exp(neg_abs_logits))

        adv_loss = crossentropy_loss(fake_output[-1]) - fake_output[-1]
        adv_loss = tf.math.reduce_mean(adv_loss)
        if self.loss_config["adv_loss"] > 0:
            gen_loss.append(
                self.loss_config["adv_loss"] * t_balance2_cond * adv_loss)

        discr_fake_loss = crossentropy_loss(fake_output[-1])
        discr_fake_loss = tf.math.reduce_mean(discr_fake_loss)
        if self.loss_config["discr_fake_loss"] > 0:
            discr_loss.append(
                self.loss_config["discr_fake_loss"] * discr_fake_loss)

        discr_real_loss = crossentropy_loss(real_output[-1]) - real_output[-1]
        discr_real_loss = tf.math.reduce_mean(discr_real_loss)
        if self.loss_config["discr_real_loss"] > 0:
            discr_loss.append(
                self.loss_config["discr_real_loss"] * discr_real_loss)

        discr_layer_loss = []
        discr_layer_norms = self.loss_config["discr_layer_norms"]
        for real_layer, fake_layer, norm in zip(
            real_output[:-1],
            fake_output[:-1],
            discr_layer_norms
        ):
            layer_loss = tf.abs(real_layer - fake_layer)
            layer_loss = tf.math.reduce_sum(layer_loss, axis=-1)
            layer_loss = tf.math.reduce_mean(layer_loss)
            discr_layer_loss.append(layer_loss / norm)
        discr_layer_loss = tf.add_n(discr_layer_loss)
        if self.loss_config["discr_layer_loss"] > 0:
            gen_loss.append(
                self.loss_config["discr_layer_loss"] * discr_layer_loss)

        vgg_loss = []
        for vgg_real, vgg_fake in zip(vgg_real_output, vgg_fake_output):
            vgg_real = tf.math.l2_normalize(vgg_real, axis=-1)
            vgg_fake = tf.math.l2_normalize(vgg_fake, axis=-1)
            cos_diff = vgg_real * vgg_fake
            cos_diff = tf.math.reduce_sum(cos_diff, axis=-1)
            cos_diff = 1 - tf.math.reduce_mean(cos_diff)
            vgg_loss.append(cos_diff)
        vgg_loss = tf.add_n(vgg_loss)
        if self.loss_config["vgg_loss"] > 0:
            gen_loss.append(self.loss_config["vgg_loss"] * vgg_loss)

        gen_loss = tf.add_n(gen_loss + self.losses)
        discr_loss = tf.add_n(discr_loss + self.losses)
        t_balance1 = adv_loss - discr_real_loss
        t_balance2 = adv_loss - discr_fake_loss

        self.content_loss_tr.update_state(
            content_loss, sample_weight=batch_size * 19)
        self.warp_loss_tr.update_state(
            warp_loss, sample_weight=batch_size * 18)
        self.vgg_loss_tr.update_state(vgg_loss, sample_weight=batch_size * 19)
        self.pp_loss_tr.update_state(pp_loss, sample_weight=batch_size * 9)
        self.adv_loss_tr.update_state(adv_loss, sample_weight=batch_size * 6)
        self.discr_layer_loss_tr.update_state(
            discr_layer_loss, sample_weight=batch_size * 6)
        self.discr_real_loss_tr.update_state(
            discr_real_loss, sample_weight=batch_size * 6)
        self.discr_fake_loss_tr.update_state(
            discr_fake_loss, sample_weight=batch_size * 6)

        return {
            "gen_loss": gen_loss / num_replicas,
            "discr_loss": discr_loss / num_replicas,
            "t_balance1": t_balance1,
            "t_balance2": t_balance2,
        }

    def compute_metrics(self, x, y, y_pred, sample_weight):
        """Compute metrics."""
        assert y is None
        assert sample_weight is None
        fake_output = y_pred["fake_output"]
        real_output = y_pred["real_output"]

        self.discr_real_acc_tr.update_state(
            tf.ones_like(real_output[-1]), tf.math.sigmoid(real_output[-1]))
        self.discr_fake_acc_tr.update_state(
            tf.zeros_like(fake_output[-1]), tf.math.sigmoid(fake_output[-1]))
        metrics = super().compute_metrics(x, y, y_pred, sample_weight)
        return {
            **metrics,
            "t_balance1": self.t_balance1_avg.result(),
            "t_balance2": self.t_balance2_avg.result(),
        }

    def train_step(self, data):
        """Train step."""
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)

        internal_layer = self.get_layer("internal")
        generator_model = internal_layer.generator_model
        flow_model = internal_layer.flow_model
        discriminator_model = internal_layer.discriminator_model
        gen_variables = generator_model.trainable_variables
        gen_variables += flow_model.trainable_variables
        discr_variables = discriminator_model.trainable_variables

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        [gen_grad, discr_grad] = tape.gradient(
            [loss["gen_loss"], loss["discr_loss"]],
            [
                gen_variables,
                discr_variables,
            ]
        )
        self.optimizer_gen.apply_gradients(zip(gen_grad, gen_variables))

        def train_discr():
            self.discr_steps_tr.update_state()
            self.optimizer_discr.apply_gradients(
                zip(discr_grad, discr_variables))
            return tf.no_op()

        self.t_balance1_avg.update_state(loss["t_balance1"])
        self.t_balance2_avg.update_state(loss["t_balance2"])

        if self.loss_config["t_balance1_threshold"] is not None:
            # Control-flow: assuming everything is in sync between all replicas
            tf.cond(
                pred=tf.less(self.t_balance1_avg.result(),
                             self.loss_config["t_balance1_threshold"]),
                true_fn=train_discr,
                false_fn=tf.no_op,
            )
        else:
            train_discr()

        # Manually advance iteration counter
        self.optimizer.iterations.assign_add(1)

        return self.compute_metrics(x, y, y_pred, sample_weight)

    def register_optimizer_variables(self):
        """Register optimizer variables."""
        with self.distribute_strategy.scope():
            internal_layer = self.get_layer("internal")
            generator_model = internal_layer.generator_model
            flow_model = internal_layer.flow_model
            discriminator_model = internal_layer.discriminator_model
            gen_variables = generator_model.trainable_variables
            gen_variables += flow_model.trainable_variables
            discr_variables = discriminator_model.trainable_variables
            self.optimizer_gen.build(gen_variables)
            self.optimizer_discr.build(discr_variables)

    def save_own_variables(self, store):
        """Save persistent variables."""
        variables = self.t_balance1_avg.variables
        variables += self.t_balance2_avg.variables
        for i, v in enumerate(variables):
            store[str(i)] = v.numpy()

    def load_own_variables(self, store):
        """Load persistent variables."""
        variables = self.t_balance1_avg.variables
        variables += self.t_balance2_avg.variables
        if len(store.keys()) == len(variables):
            for i, v in enumerate(variables):
                v.assign(store[str(i)])
        else:
            warnings.warn("GAN variables are not loaded")

    def get_config(self) -> Dict[str, Any]:
        """Get model config.

        Returns
        -------
        Dict[str, Any]
            Model config
        """
        config = super().get_config()
        return {
            **config,
            "loss_config": self.loss_config,
        }

    @staticmethod
    def _get_loss_config(loss_config: Union[None, Dict[str, Any]] = None) \
            -> Dict[str, Any]:
        """Create loss config."""
        if loss_config is None or not isinstance(loss_config, dict):
            loss_config = {}
        loss_config = {
            "content_loss": 1.0,
            "pp_loss": 0.5,
            "warp_loss": 1.0,
            "adv_loss": 0.1,
            "discr_layer_norms": [12.0, 14.0, 48.0, 250.0],
            "discr_layer_loss": 0.2,
            "vgg_loss": 0.2,
            "discr_real_loss": 1.0,
            "discr_fake_loss": 1.0,
            "t_balance1_threshold": 0.2,
            "t_balance2_threshold": 0.0,
            **loss_config
        }
        return loss_config

    @staticmethod
    def _build_model_args(
        generator_model: keras.Model,
        flow_model: keras.Model,
        discriminator_model: keras.Model,
        vgg_model: keras.Model,
        crop_size: int,
    ):
        inputs = keras.Input(shape=[10, crop_size, crop_size, 3],
                             name="input", dtype="float32")
        targets = keras.Input(shape=[10, crop_size*4, crop_size*4, 3],
                              name="target", dtype="float32")

        class InternalLayer(layers.Layer):
            """Internal layer."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.generator_model = generator_model
                self.flow_model = flow_model
                self.discriminator_model = discriminator_model
                self.vgg_model = vgg_model

            def call(self, inputs, targets):
                """Execute training inference."""
                input_shape = tf.shape(inputs[:, 0, :, :, :])
                output_shape = tf.shape(targets[:, 0, :, :, :])
                inputs_rev = inputs[:, -2::-1, :, :, :]
                targets_rev = targets[:, -2::-1, :, :, :]
                inputs_d = tf.concat([inputs, inputs_rev], axis=1)
                targets_d = tf.concat([targets, targets_rev], axis=1)
                input_frames = tf.reshape(inputs_d[:, 1:, :, :, :],
                                          [-1, crop_size, crop_size, 3])
                input_frames_pre = tf.reshape(inputs_d[:, :-1, :, :, :],
                                              [-1, crop_size, crop_size, 3])
                num_rand_frames = len(flow_model.inputs) - 2
                if num_rand_frames > 0:
                    # Random last frames for flow generation
                    rand_frames = tf.reshape(
                        tf.stack([
                            tf.random.uniform(
                                input_shape,
                                minval=-0.5,
                                maxval=0.5,
                                dtype=inputs.dtype
                            )
                            for _ in range(num_rand_frames)
                        ], axis=1),
                        [-1, num_rand_frames, crop_size, crop_size, 3]
                    )
                flow_last_frames = [
                    tf.reshape(
                        tf.concat([
                            rand_frames[:, -(i + 1):, :, :, :],
                            inputs_d[:, :-(i + 2), :, :, :]
                        ], axis=1),
                        [-1, crop_size, crop_size, 3]
                    )
                    for i in range(num_rand_frames)
                ]
                target_frames_pre = tf.reshape(
                    targets_d[:, :-1, :, :, :],
                    [-1, crop_size*4, crop_size*4, 3]
                )
                flow = flow_model(
                    [input_frames, input_frames_pre] + flow_last_frames)
                target_warp = DenseWarpLayer()([target_frames_pre, flow])
                target_warp = tf.reshape(
                    target_warp, [-1, 18, crop_size*4, crop_size*4, 3])
                flow = tf.reshape(flow, [-1, 18, crop_size*4, crop_size*4, 2])
                # First frame uses random pre_warp
                last_output = generator_model([
                    inputs[:, 0, :, :, :],
                    tf.random.uniform(output_shape, minval=-0.5,
                                      maxval=0.5, dtype=inputs.dtype)
                ])
                gen_outputs = [last_output]
                gen_warp = []
                for frame_i in range(18):
                    cur_flow = flow[:, frame_i, :, :, :]
                    gen_pre_output_warp = DenseWarpLayer()(
                        [last_output, cur_flow])
                    last_output = generator_model([
                        inputs_d[:, frame_i + 1, :, :, :],
                        gen_pre_output_warp
                    ])
                    gen_outputs.append(last_output)
                    gen_warp.append(gen_pre_output_warp)
                gen_outputs = tf.reshape(
                    tf.stack(gen_outputs, axis=1),
                    [-1, 19, crop_size*4, crop_size*4, 3]
                )
                gen_warp = tf.reshape(
                    tf.stack(gen_warp, axis=1),
                    [-1, 18, crop_size*4, crop_size*4, 3]
                )
                vgg_real_output = vgg_model(
                    tf.reshape(targets, [-1, crop_size * 4, crop_size * 4, 3]),
                    training=False
                )
                vgg_real_output = [
                    tf.reshape(x, [-1, 10] + list(out.shape)[1:])
                    for x, out in zip(vgg_real_output, vgg_model.outputs)
                ]
                vgg_real_output = [
                    tf.concat([x, x[:, -2::-1]], axis=1)
                    for x in vgg_real_output
                ]
                vgg_fake_output = vgg_model(
                    tf.reshape(
                        gen_outputs, [-1, crop_size * 4, crop_size * 4, 3]),
                    training=False
                )
                vgg_fake_output = [
                    tf.reshape(x, [-1, 19] + list(out.shape)[1:])
                    for x, out in zip(vgg_fake_output, vgg_model.outputs)
                ]
                t_gen_outputs = tf.reshape(gen_outputs[:, :18, :, :, :],
                                           [-1, crop_size*4, crop_size*4, 3])
                t_targets = tf.reshape(targets_d[:, :18, :, :, :],
                                       [-1, crop_size*4, crop_size*4, 3])
                t_inputs = tf.reshape(inputs_d[:, :18, :, :, :],
                                      [-1, crop_size, crop_size, 3])
                inputs_hi = UpscaleLayer(scale=4)(t_inputs)
                inputs_hi = tf.reshape(
                    inputs_hi, [-1, 3, crop_size*4, crop_size*4, 3])
                inputs_hi = tf.transpose(inputs_hi, [0, 2, 3, 4, 1])
                inputs_hi = tf.reshape(
                    inputs_hi, [-1, crop_size*4, crop_size*4, 9])
                t_inputs_vpre_batch = flow[:, :18:3, :, :, :]
                t_inputs_v_batch = tf.zeros_like(t_inputs_vpre_batch)
                t_inputs_vnxt_batch = flow[:, -2:-19:-3, :, :, :]
                t_vel = tf.stack([t_inputs_vpre_batch, t_inputs_v_batch,
                                  t_inputs_vnxt_batch], axis=2)
                t_vel = tf.reshape(t_vel, [-1, crop_size*4, crop_size*4, 2])
                t_vel = tf.stop_gradient(t_vel)

                def get_warp(inputs):
                    warp = DenseWarpLayer()([inputs, t_vel])
                    warp = tf.reshape(
                        warp, [-1, 3, crop_size*4, crop_size*4, 3])
                    warp = tf.transpose(warp, [0, 2, 3, 4, 1])
                    warp = tf.reshape(warp, [-1, crop_size*4, crop_size*4, 9])
                    work_size = int(crop_size * 4 * 0.75)
                    pad_size = crop_size * 2 - work_size // 2
                    warp = warp[:, pad_size:pad_size+work_size,
                                pad_size:pad_size+work_size, :]
                    warp = tf.pad(warp, [[0, 0], [pad_size, pad_size],
                                         [pad_size, pad_size], [0, 0]],
                                  "CONSTANT")
                    before_warp = tf.reshape(
                        inputs, [-1, 3, crop_size*4, crop_size*4, 3])
                    before_warp = tf.transpose(before_warp, [0, 2, 3, 4, 1])
                    before_warp = tf.reshape(before_warp,
                                             [-1, crop_size*4, crop_size*4, 9])
                    warp = layers.Concatenate()([before_warp, warp, inputs_hi])
                    return warp

                real_warp = get_warp(t_targets)
                real_output = discriminator_model(real_warp)
                real_output = [
                    tf.reshape(x, [-1, 6] + list(out.shape)[1:])
                    for x, out in zip(real_output, discriminator_model.outputs)
                ]
                fake_warp = get_warp(t_gen_outputs)
                fake_output = discriminator_model(fake_warp)
                fake_output = [
                    tf.reshape(x, [-1, 6] + list(out.shape)[1:])
                    for x, out in zip(fake_output, discriminator_model.outputs)
                ]
                return [gen_outputs,
                        gen_warp,
                        target_warp,
                        vgg_real_output,
                        vgg_fake_output,
                        real_output,
                        fake_output]

        [gen_outputs,
         gen_warp,
         target_warp,
         vgg_real_output,
         vgg_fake_output,
         real_output,
         fake_output] = InternalLayer(name="internal")(inputs, targets)

        gen_outputs = layers.Identity(name="gen_outputs",
                                      dtype="float32")(gen_outputs)
        gen_warp = layers.Identity(name="gen_warp",
                                   dtype="float32")(gen_warp)
        target_warp = layers.Identity(
            name="target_warp", dtype="float32")(target_warp)
        vgg_real_output = layers.Identity(name="vgg_real_output",
                                          dtype="float32")(vgg_real_output)
        vgg_fake_output = layers.Identity(name="vgg_fake_output",
                                          dtype="float32")(vgg_fake_output)
        real_output = layers.Identity(name="real_output",
                                      dtype="float32")(real_output)
        fake_output = layers.Identity(name="fake_output",
                                      dtype="float32")(fake_output)
        return {
            "inputs": [inputs, targets],
            "outputs": {
                "gen_outputs": gen_outputs,
                "gen_warp": gen_warp,
                "target_warp": target_warp,
                "real_output": real_output,
                "fake_output": fake_output,
                "vgg_real_output": vgg_real_output,
                "vgg_fake_output": vgg_fake_output,
            }
        }
