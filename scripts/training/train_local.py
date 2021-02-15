#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Local training."""

import os
import sys
import argparse
import json
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import dataset
import utils
import training
import config


def parse_args():
    """
    Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Local training for JoshUpsale")
    parser.add_argument("-d", "--dataset-path",
                        help="Path to dataset",
                        type=str,
                        required=True)
    parser.add_argument("-t", "--tag",
                        help="Run tag",
                        type=str,
                        required=False)
    parser.add_argument("-g", "--gpu",
                        dest="gpus",
                        help="Visible GPUs",
                        type=str,
                        nargs="+",
                        required=False)
    parser.add_argument("-b", "--batch-size",
                        help="Batch size (default: %(default)d)",
                        type=int,
                        default=64,
                        required=False)
    parser.add_argument("--test-set-size",
                        dest="num_test_set",
                        help=("Number of validation examples " +
                              "(default: %(default)d)"),
                        type=int,
                        default=200,
                        required=False)
    parser.add_argument("--play-set-size",
                        dest="num_play_set",
                        help=("Number of prediction examples " +
                              "(default: %(default)d)"),
                        type=int,
                        default=8,
                        required=False)
    parser.add_argument("-c", "--crop",
                        dest="crop_size",
                        help="Crop size (default: %(default)d)",
                        type=int,
                        default=32,
                        required=False)
    parser.add_argument("-m", "--model",
                        dest="model_type",
                        help="Model type",
                        choices=["large"],
                        type=str,
                        default=None,
                        required=False)
    parser.add_argument("--config",
                        dest="config_override",
                        help="Path to model config override file",
                        type=str,
                        default=None,
                        required=False)
    parser.add_argument("--skip-frvsr",
                        dest="frvsr_skip",
                        help="Skip FRVSR training",
                        action="store_true",
                        default=False,
                        required=False)
    parser.add_argument("--frvsr-epochs",
                        help=("Number of epochs for FRVSR training " +
                              "(default: %(default)d)"),
                        type=int,
                        default=400,
                        required=False)
    parser.add_argument("--frvsr-steps",
                        help=("Number of steps prt epoch for FRVSR training " +
                              "(default: %(default)d)"),
                        type=int,
                        default=100,
                        required=False)
    parser.add_argument("--gan-epochs",
                        help=("Number of epochs for GAN training " +
                              "(default: %(default)d)"),
                        type=int,
                        default=400,
                        required=False)
    parser.add_argument("--gan-steps",
                        help=("Number of steps prt epoch for GAN training " +
                              "(default: %(default)d)"),
                        type=int,
                        default=100,
                        required=False)
    parser.add_argument("--logdir",
                        dest="tensorboard_dir",
                        help="TensorBoard log directory",
                        type=str,
                        default=None,
                        required=False)
    parser.add_argument("--checkpointdir",
                        dest="checkpoint_dir",
                        help="Checkpoint directory",
                        type=str,
                        default=None,
                        required=False)
    parser.add_argument("--seed",
                        dest="random_seed",
                        help="Random seed",
                        type=int,
                        default=None,
                        required=False)

    return parser.parse_args()


def init(
    gpus=None,
    random_seed=None
):
    """
    Init hardware and libraries.

    Parameters
    ----------
    gpus : list of strings
        Visible GPUs
    random_seed : int
        Random seed

    Returns
    -------
    tf.distribute.Strategy
        Distributed training strategy if any
    """
    strategy = None
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        details = tf.config.experimental.get_device_details(device)
        compute_capability = details.get("compute_capability", None)
        if compute_capability and compute_capability[0] >= 7:
            keras.mixed_precision.experimental.set_policy("mixed_float16")
    tf.config.optimizer.set_jit(True)
    if random_seed is not None:
        utils.set_random_seed(random_seed)
    if len(physical_devices) > 1:
        strategy = tf.distribute.MirroredStrategy(
            devices=[device.name for device in physical_devices]
        )
    return strategy


def load_datasets(
    dataset_path,
    batch_size=64,
    num_test_set=200,
    num_play_set=8,
    crop_size=32
):
    """
    Load datasets.

    Parameters
    ----------
    dataset_path : str
        Dataset path
    batch_size : int
        Batch size
    num_test_set : int
        Number of test examples
    num_play_set : int
        Number of play examples
    crop_size : int
        Cropping size

    Returns
    -------
    tf.Dataset
        Train dataset
    tf.Dataset
        Validation dataset
    tf.Dataset
        PLay dataset (for visualisation)
    """
    train_ds, val_ds = dataset.get_dataset(
        path=dataset_path,
        batch_size=batch_size,
        num_test_set=num_test_set,
        crop_size=crop_size
    )
    play_ds = val_ds.unbatch().take(num_play_set)
    play_ds = play_ds.batch(
        min(num_play_set, batch_size),
        drop_remainder=True
    )
    play_ds = play_ds.cache()
    val_ds = val_ds.cache()
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, play_ds


def train_frvsr(
    model,
    train_ds,
    val_ds,
    play_ds,
    epochs=400,
    steps=100,
    tag=None,
    logdir_base=None,
    checkpointdir_base=None
):
    """
    Train FRVSR.

    Parameters
    ----------
    model : training.Training or training.DistributedTraining
        Training model
    train_ds : tf.Dataset
        Train dataset
    val_ds : tf.Dataset
        Validation dataset
    play_ds : tf.Dataset
        Play dataset (for visualisation)
    epochs : int
        Number of epochs
    steps : int
        Number of steps per epoch
    tag : str
        Run tag
    logdir_base : str
        Base directory for tensorboard logs
    checkpointdir_base : str
        Base directory for checkpoints
    """
    initial_epoch = 0
    callbacks = []
    if logdir_base is not None:
        logdir = os.path.join(logdir_base, "frvsr")
        if tag is not None:
            logdir = os.path.join(logdir, tag)
        callbacks.append(
            keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
        )
        metrics_writer = tf.summary.create_file_writer(
            os.path.join(logdir, "metrics")
        )

        def metrics_fn(epoch, _logs):
            # pylint: disable=not-context-manager
            data = model.play(play_ds)
            with metrics_writer.as_default():
                utils.gif_summary(
                    name="gen_outputs",
                    tensor=tf.convert_to_tensor(data[0]),
                    step=epoch
                )
                utils.gif_summary(
                    name="pre_warps",
                    tensor=tf.convert_to_tensor(data[1]),
                    step=epoch
                )
            metrics_writer.flush()
        callbacks.append(
            keras.callbacks.LambdaCallback(on_epoch_end=metrics_fn)
        )
    if checkpointdir_base is not None:
        checkpointdir = os.path.join(checkpointdir_base, "frvsr")
        if tag is not None:
            checkpointdir = os.path.join(checkpointdir, tag)
        checkpoint_format = \
            os.path.join(checkpointdir, "weights-{epoch:03d}.h5")
        if os.path.exists(checkpointdir):
            for i in range(epochs, 0, -1):
                filename = checkpoint_format.format(epoch=i)
                if os.path.exists(filename):
                    initial_epoch = i
                    model.frvsr_model.load_weights(filename)
                    break
        else:
            os.makedirs(checkpointdir)
        callbacks.append(keras.callbacks.ModelCheckpoint(
            checkpoint_format, save_weights_only=True
        ))
        callbacks.append(keras.callbacks.ModelCheckpoint(
            os.path.join(checkpointdir, "weights-best.h5"),
            monitor="val_gen_outputs_loss",
            save_best_only=True,
            save_weights_only=True
        ))
    model.train_frvsr(
        train_ds,
        epochs=epochs,
        steps=steps,
        validation_data=val_ds,
        callbacks=callbacks,
        initial_epoch=initial_epoch
    )


def main(
    dataset_path,
    tag=None,
    gpus=None,
    batch_size=64,
    num_test_set=200,
    num_play_set=8,
    crop_size=32,
    model_type="large",
    config_override=None,
    frvsr_skip=False,
    frvsr_epochs=400,
    frvsr_steps=100,
    gan_epochs=400,  # pylint: disable=unused-argument
    gan_steps=100,  # pylint: disable=unused-argument
    tensorboard_dir=None,
    checkpoint_dir=None,
    random_seed=None
):
    """
    Run CLI.

    Parameters
    ----------
    dataset_path : str
        Dataset directory
    tag : str
        Run tag
    gpus : list of str
        Visible GPUs
    batch_size : int
        Batch size
    num_test_set : int
        Number of test examples for validation
    num_play_set : int
        Number of test examples for visualisation
    crop_size : int
        Cropping size
    model_type : str
        Model type
    config_override : str or dict
        Model config override
    frvsr_skip : bool
        Skip FRVSR training
    frvsr_epochs : int
        Number of epochs for FRVSR training
    frvsr_steps : int
        Number of steps per epoch for FRVSR training
    gan_epochs : int
        Number of epochs for GAN training
    gan_steps : int
        Number of steps per epoch for GAN training
    tensorboard_dir : str
        Path to tensorboard log directory
    checkpoint_dir : tr
        Path to checkpoint directory
    random_seed : int
        Random seed

    Returns
    -------
    int
        Exit code
    """
    if tag is None:
        tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    strategy = init(gpus=gpus, random_seed=random_seed)
    train_ds, val_ds, play_ds = load_datasets(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_test_set=num_test_set,
        num_play_set=num_play_set,
        crop_size=crop_size
    )
    if config_override is not None and isinstance(config_override, str):
        with open(config_override, "rt") as config_file:
            config_override = json.load(config_file)
    model_config = config.get_config(
        model=model_type,
        batch_size=batch_size,
        crop_size=crop_size,
        learning_rate=0.0005,
        config_override=config_override
    )
    if strategy is None:
        model = training.Training(model_config)
    else:
        model = training.DistributedTraining(model_config, strategy)
    model.init()
    if not frvsr_skip:
        train_frvsr(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            play_ds=play_ds,
            epochs=frvsr_epochs,
            steps=frvsr_steps,
            logdir_base=tensorboard_dir,
            checkpointdir_base=checkpoint_dir,
            tag=tag
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(**vars(parse_args())))
    except KeyboardInterrupt:
        sys.exit(-1)
