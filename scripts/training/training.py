# -*- coding: utf-8 -*-

"""Training routines."""

import sys
import tensorflow as tf
from tensorflow import keras
import tqdm
import model


class Training:
    """Training class."""

    def __init__(self, config):
        """
        Construct Training object.

        Parameters
        ----------
        config : dict
            Training config
        """
        self.config = config
        self.generator_model = None
        self.flow_model = None
        self.full_model = None
        self.frvsr_model = None
        self.discriminator_model = None
        self.gan_model = None
        self.generator_optimizer = None
        self.flow_optimizer = None
        self.discriminator_optimizer = None
        self.steps_per_execution = None
        self.avg_gen_loss = None
        self.avg_gen_loss = None
        self.avg_fnet_loss = None
        self.avg_discr_loss = None
        self.avg_content_loss = None
        self.avg_warp_loss = None
        self.step_fn = None
        self.test_fn = None
        self.play_fn = None
        self.multi_step_fn = None
        self.multi_test_fn = None
        self.multi_play_fn = None

    def init(self):
        """Initialise models."""
        generator_config = self.config["generator"]
        self.generator_model = model.get_generator_model(
            **generator_config
        )
        flow_config = self.config["flow"]
        if self.config["flow_model_type"] == "resnet":
            self.flow_model = model.get_flow_model_resnet(
                **flow_config
            )
        elif self.config["flow_model_type"] == "autoencoder":
            self.flow_model = model.get_flow_model_autoencoder(
                **flow_config
            )
        else:
            raise ValueError("Unknown flow model type: {}".format(
                flow_config["type"]))
        full_model_config = self.config["full_model"]
        self.full_model = model.get_full_model(
            generator_model=self.generator_model,
            flow_model=self.flow_model,
            **full_model_config
        )
        frvsr_config = self.config["frvsr"]
        self.frvsr_model = model.get_frvsr(
            generator_model=self.generator_model,
            flow_model=self.flow_model,
            **frvsr_config
        )
        discriminator_config = self.config["discriminator"]
        self.discriminator_model = model.get_discriminator_model(
            **discriminator_config
        )
        gan_config = self.config["gan"]
        self.gan_model = model.get_gan_model(
            generator_model=self.generator_model,
            flow_model=self.flow_model,
            discriminator_model=self.discriminator_model,
            **gan_config
        )
        gan_train_config = self.config["gan_train"]
        self.generator_optimizer = keras.optimizers.Adam(
            learning_rate=gan_train_config["generator_learning_rate"]
        )
        self.flow_optimizer = keras.optimizers.Adam(
            learning_rate=gan_train_config["flow_learning_rate"]
        )
        self.discriminator_optimizer = keras.optimizers.Adam(
            learning_rate=gan_train_config["discriminator_learning_rate"]
        )
        self.steps_per_execution = gan_train_config["steps_per_execution"]

        self.avg_gen_loss = keras.metrics.Mean("gen_loss")
        self.avg_fnet_loss = keras.metrics.Mean("fnet_loss")
        self.avg_discr_loss = keras.metrics.Mean("discr_loss")
        self.avg_content_loss = keras.metrics.Mean("content_loss")
        self.avg_warp_loss = keras.metrics.Mean("warp_loss")

        @tf.function
        def get_loss(targets, gen_outputs, target_warp, fake_output,
                     real_output, warp_outputs):
            # pylint: disable=unexpected-keyword-arg
            # pylint: disable=no-value-for-parameter
            # pylint: disable=invalid-unary-operand-type
            targets_b = tf.concat([targets, targets[:, -2::-1, :, :, :]],
                                  axis=1)
            targets_t = targets_b[:, 1:, :, :, :]

            content_loss = gen_outputs - targets_b
            content_loss = tf.square(content_loss)
            content_loss = tf.reduce_sum(content_loss, axis=[4])
            content_loss = tf.reduce_mean(
                content_loss,
                axis=range(1, len(content_loss.shape))
            )

            warp_cont_loss = gen_outputs[:, 1:, :, :, :] - warp_outputs
            warp_cont_loss = tf.square(warp_cont_loss)
            warp_cont_loss = tf.reduce_sum(warp_cont_loss, axis=[4])
            warp_cont_loss = tf.reduce_mean(
                warp_cont_loss,
                axis=range(1, len(warp_cont_loss.shape))
            )

            warp_loss = target_warp - targets_t
            warp_loss = tf.square(warp_loss)
            warp_loss = tf.reduce_sum(warp_loss, axis=[4])
            warp_loss = tf.reduce_mean(
                warp_loss,
                axis=range(1, len(warp_loss.shape))
            )

            gen_out_first = gen_outputs[:, :9, :, :, :]
            gen_out_last_rev = gen_outputs[:, -1:-10:-1, :, :, :]
            pp_loss = gen_out_first - gen_out_last_rev
            pp_loss = tf.abs(pp_loss)
            pp_loss = tf.reduce_mean(
                pp_loss,
                axis=range(1, len(pp_loss.shape))
            )

            t_adversarial_loss = -tf.math.log(fake_output[-1] + 0.01)
            t_adversarial_loss = tf.reduce_mean(
                t_adversarial_loss,
                axis=range(1, len(t_adversarial_loss.shape))
            )
            t_discrim_fake_loss = -tf.math.log(1 - fake_output[-1] + 0.01)
            t_discrim_fake_loss = tf.reduce_mean(
                t_discrim_fake_loss,
                axis=range(1, len(t_discrim_fake_loss.shape))
            )
            t_discrim_real_loss = -tf.math.log(real_output[-1] + 0.01)
            t_discrim_real_loss = tf.reduce_mean(
                t_discrim_real_loss,
                axis=range(1, len(t_discrim_real_loss.shape))
            )

            sum_layer_loss = 0
            layer_norm = [12.0, 14.0, 24.0, 100.0]
            for i in range(4):
                real_layer = real_output[i]
                fake_layer = fake_output[i]
                layer_loss = real_layer - fake_layer
                layer_loss = tf.abs(layer_loss)
                layer_loss = tf.reduce_sum(layer_loss, axis=[3])
                layer_loss = tf.reduce_mean(
                    layer_loss, axis=range(1, len(layer_loss.shape)))
                sum_layer_loss += 0.02 * layer_loss / layer_norm[i]

            gen_loss = content_loss + pp_loss * 0.3 + t_adversarial_loss * \
                0.01 + sum_layer_loss + warp_cont_loss * 0.001
            fnet_loss = warp_loss + content_loss + pp_loss * \
                0.3 + t_adversarial_loss * 0.01 + sum_layer_loss
            discr_loss = t_discrim_fake_loss + t_discrim_real_loss
            t_balance = t_adversarial_loss - t_discrim_real_loss

            gen_loss = tf.nn.compute_average_loss(
                gen_loss, global_batch_size=self.config["batch_size"])
            fnet_loss = tf.nn.compute_average_loss(
                fnet_loss, global_batch_size=self.config["batch_size"])
            discr_loss = tf.nn.compute_average_loss(
                discr_loss, global_batch_size=self.config["batch_size"])
            content_loss = tf.nn.compute_average_loss(
                content_loss, global_batch_size=self.config["batch_size"])
            warp_loss = tf.nn.compute_average_loss(
                warp_loss, global_batch_size=self.config["batch_size"])
            t_balance = tf.reduce_mean(t_balance)
            t_balance = tf.stop_gradient(t_balance)

            return (gen_loss, fnet_loss, discr_loss, t_balance, content_loss,
                    warp_loss)

        @tf.function
        def step_fn(inputs, targets):
            with tf.GradientTape() as tape:
                gen_outputs, target_warp, fake_output, real_output, \
                    warp_outputs = self.gan_model([inputs, targets])
                gen_loss, fnet_loss, discr_loss, t_balance, content_loss, \
                    warp_loss = get_loss(targets, gen_outputs, target_warp,
                                         fake_output, real_output,
                                         warp_outputs)
                cond_discr_loss = (1 - tf.math.sign(t_balance - 0.4)) / \
                    2 * discr_loss
            [gen_grad, fnet_grad, discr_grad] = tape.gradient(
                [gen_loss, fnet_loss, cond_discr_loss],
                [
                    self.generator_model.trainable_variables,
                    self.flow_model.trainable_variables,
                    self.discriminator_model.trainable_variables
                ]
            )
            self.generator_optimizer.apply_gradients(
                zip(gen_grad, self.generator_model.trainable_variables))
            self.flow_optimizer.apply_gradients(
                zip(fnet_grad, self.flow_model.trainable_variables))
            self.discriminator_optimizer.apply_gradients(
                zip(discr_grad, self.discriminator_model.trainable_variables))
            self.avg_gen_loss.update_state(gen_loss)
            self.avg_fnet_loss.update_state(fnet_loss)
            self.avg_discr_loss.update_state(discr_loss)
            self.avg_content_loss.update_state(content_loss)
            self.avg_warp_loss.update_state(warp_loss)
            return gen_loss, fnet_loss, discr_loss

        @tf.function
        def test_fn(inputs, targets):
            gen_outputs, input_warp, fake_output, real_output, warp_outputs = \
                self.gan_model([inputs, targets], training=False)
            gen_loss, fnet_loss, discr_loss, _, content_loss, \
                warp_loss = get_loss(targets, gen_outputs, input_warp,
                                     fake_output, real_output, warp_outputs)
            self.avg_gen_loss.update_state(gen_loss)
            self.avg_fnet_loss.update_state(fnet_loss)
            self.avg_discr_loss.update_state(discr_loss)
            self.avg_content_loss.update_state(content_loss)
            self.avg_warp_loss.update_state(warp_loss)

        @tf.function
        def play_fn(inputs):
            shape = tf.shape(inputs)
            batch_size = shape[0]
            height = shape[2]
            width = shape[3]
            last_frame = tf.zeros((batch_size, height, width, 3))
            last_output = tf.zeros((batch_size, height*2, width*2, 3))
            gen_outputs = []
            pre_warps = []
            for i in range(10):
                cur_frame = inputs[:, i, :, :, :]
                last_output, pre_warp = self.full_model(
                    [cur_frame, last_frame, last_output],
                    training=False
                )
                last_frame = cur_frame
                gen_outputs.append(last_output)
                pre_warps.append(pre_warp)
            gen_outputs = tf.reshape(tf.stack(gen_outputs, axis=1),
                                     [-1, 10, height*2, width*2, 3])
            pre_warps = tf.reshape(tf.stack(pre_warps, axis=1),
                                   [-1, 10, height*2, width*2, 3])
            return gen_outputs, pre_warps

        @tf.function
        def multi_step_fn(ds_iter):
            for _ in tf.range(self.steps_per_execution):
                data = next(ds_iter)
                gen_loss, fnet_loss, discr_loss = self.step_fn(
                    data["input"], data["target"])
            return gen_loss, fnet_loss, discr_loss

        def multi_test_fn(dataset):
            for data in dataset:
                self.test_fn(data["input"], data["target"])

        def multi_play_fn(dataset):
            states = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            for data in dataset:
                gen_outputs, pre_warps = self.play_fn(data["input"])
                images = tf.stack([gen_outputs, pre_warps], axis=1)
                states = states.write(states.size(), images)
            return tf.transpose(states.concat(), [1, 0, 2, 3, 4, 5])

        self.step_fn = step_fn
        self.test_fn = test_fn
        self.play_fn = play_fn
        self.multi_step_fn = multi_step_fn
        self.multi_test_fn = multi_test_fn
        self.multi_play_fn = multi_play_fn

        if gan_train_config["compile_test_fn"]:
            self.multi_test_fn = tf.function(self.multi_test_fn)
        if gan_train_config["compile_play_fn"]:
            self.multi_play_fn = tf.function(self.multi_play_fn)

    def train_frvsr(self, train_data, epochs, steps, initial_epoch=0,
                    validation_data=None, callbacks=None):
        """
        Train FRVSR model.

        Parameters
        ----------
        train_data : tf.data.Dataset
            Train dataset
        epochs : int
            Number of epochs
        steps : int
            Number of steps per epoch
        initial_epoch : int
            First epoch
        validation_data : tf.data.Dataset
            Validation dataset
        callbacks : array of keras.callbacks.Callback
            Callbacks
        """
        def frvsr_map(val):
            return (
                (val["input"], val["target"]),
                {
                    "target_warp": val["target"][:, 1:, :, :, :],
                    "gen_outputs": val["target"]
                }
            )
        self.frvsr_model.fit(
            train_data.map(frvsr_map),
            epochs=epochs,
            steps_per_epoch=steps,
            initial_epoch=initial_epoch,
            validation_data=validation_data.map(
                frvsr_map) if validation_data is not None else None,
            validation_steps=(len([x for x in validation_data])
                              if validation_data is not None else None),
            callbacks=callbacks,
        )

    def train_gan(self, train_data, epochs, steps, initial_epoch=0,
                  validation_data=None, callbacks=None):
        """
        Train GAN model.

        Parameters
        ----------
        train_data : tf.data.Dataset
            Train dataset
        epochs : int
            Number of epochs
        steps : int
            Number of steps per epoch
        initial_epoch : int
            First epoch
        validation_data : tf.data.Dataset
            Validation dataset
        callbacks : func
            Callbacks
        """
        bar_format = "{n_fmt}/{total_fmt} [{bar:35}] {elapsed_s:.0f}s," + \
            "{rate_fmt}{postfix}"
        ds_iter = iter(train_data)

        for epoch in range(initial_epoch, epochs):
            print("Epoch %d/%d" % (epoch + 1, epochs))
            self.avg_gen_loss.reset_states()
            self.avg_fnet_loss.reset_states()
            self.avg_discr_loss.reset_states()
            self.avg_content_loss.reset_states()
            self.avg_warp_loss.reset_states()
            with tqdm.tqdm(range(0, steps, self.steps_per_execution),
                           unit_scale=self.steps_per_execution, unit="step",
                           file=sys.stdout, bar_format=bar_format,
                           ascii=".=") as step_progress:
                for _ in step_progress:
                    gen_loss, fnet_loss, discr_loss = self.multi_step_fn(
                        ds_iter)
                    step_progress.set_postfix({
                        "gen_loss": gen_loss.numpy(),
                        "fnet_loss": fnet_loss.numpy(),
                        "discr_loss": discr_loss.numpy()
                    })
                step_progress.set_postfix({
                    "gen_loss": self.avg_gen_loss.result().numpy(),
                    "fnet_loss": self.avg_fnet_loss.result().numpy(),
                    "discr_loss": self.avg_discr_loss.result().numpy()
                })
            metrics = {
                "gen_loss": self.avg_gen_loss.result(),
                "fnet_loss": self.avg_fnet_loss.result(),
                "discr_loss": self.avg_discr_loss.result(),
                "content_loss": self.avg_content_loss.result(),
                "warp_loss": self.avg_warp_loss.result()
            }
            if validation_data is not None:
                self.avg_gen_loss.reset_states()
                self.avg_fnet_loss.reset_states()
                self.avg_discr_loss.reset_states()
                self.avg_content_loss.reset_states()
                self.avg_warp_loss.reset_states()
                self.multi_test_fn(validation_data)
                metrics = {
                    **metrics,
                    "val_gen_loss": self.avg_gen_loss.result(),
                    "val_fnet_loss": self.avg_fnet_loss.result(),
                    "val_discr_loss": self.avg_discr_loss.result(),
                    "val_content_loss": self.avg_content_loss.result(),
                    "val_warp_loss": self.avg_warp_loss.result()
                }
                print("gen_loss", metrics["val_gen_loss"].numpy(),
                      "fnet_loss", metrics["val_fnet_loss"].numpy(),
                      "discr_loss", metrics["val_discr_loss"].numpy(),
                      "content_loss", metrics["val_content_loss"].numpy(),
                      "warp_loss", metrics["val_warp_loss"].numpy())
            if callbacks:
                for callback in callbacks:
                    callback(epoch + 1, metrics)

    def play(self, data):
        """
        Run model on dataset.

        Parameters
        ----------
        data : tf.data.Dataset
            Play dataset

        Returns
        -------
        tf.Tensor (2 x N x H x W x 3)
            Generated frame and warped frame
        """
        return self.multi_play_fn(data)


class DistributedTraining(Training):
    """Distributed training."""

    def __init__(self, config, strategy):
        """
        Construct DistributedTraining object.

        Parameters
        ----------
        config : dict
            Model config
        strategy : tf.distribute.Strategy
            Distributed strategy
        """
        super().__init__(config)
        self.strategy = strategy

    def init(self):
        """Initialise models."""
        with self.strategy.scope():
            super().init()

        step_fn = self.step_fn
        test_fn = self.test_fn
        play_fn = self.play_fn

        @tf.function
        def distributed_step_fn(inputs, targets):
            per_replica_losses = self.strategy.run(
                step_fn, args=(inputs, targets))
            out_losses = ()
            for pre_replice_loss in per_replica_losses:
                out_losses += (self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM,
                    pre_replice_loss, axis=None
                ), )
            return out_losses

        @tf.function
        def distributed_test_fn(inputs, targets):
            self.strategy.run(test_fn, args=(inputs, targets))

        @tf.function
        def distributed_play_fn(inputs):
            gen_outputs, pre_warps = self.strategy.run(
                play_fn, args=(inputs, ))
            return (self.strategy.gather(gen_outputs, axis=0),
                    self.strategy.gather(pre_warps, axis=0))

        self.step_fn = distributed_step_fn
        self.test_fn = distributed_test_fn
        self.play_fn = distributed_play_fn

    def train_gan(self, train_data, epochs, steps, initial_epoch=0,
                  validation_data=None, callbacks=None):
        """
        Train GAN model.

        Parameters
        ----------
        train_data : tf.data.Dataset
            Train dataset
        epochs : int
            NUmber of epochs
        steps : int
            Number of steps per epoch
        initial_epoch : int
            First epoch
        validation_data : tf.data.Dataset
            Validation dataset
        callbacks : func
            Callbacks
        """
        super().train_gan(
            train_data=self.strategy.experimental_distribute_dataset(
                train_data),
            epochs=epochs,
            steps=steps,
            initial_epoch=initial_epoch,
            validation_data=self.strategy.experimental_distribute_dataset(
                validation_data) if validation_data is not None else None,
            callbacks=callbacks
        )

    def play(self, data):
        """
        Run model on dataset.

        Parameters
        ----------
        data : tf.distribute.DistributedDataset
            Play dataset

        Returns
        -------
        np.darray (2 x N x H x W x 3)
            Generated frame and warped frame
        """
        return self.multi_play_fn(data)
