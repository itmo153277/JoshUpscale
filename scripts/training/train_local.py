#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run local training."""

import os
import sys
import logging
import argparse
from typing import Any, Dict, List, Union
import yaml
import tensorflow as tf
from tensorflow import keras
from models import create_models
from dataset import create_train_dataset, create_val_dataset
from keras_callbacks import PlayCallback, TensorBoard


LOG = logging.getLogger("train_local")


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run local training")
    parser.add_argument("-c", "--config",
                        help="Config",
                        type=str,
                        dest="config_path",
                        required=True)
    parser.add_argument("-g", "--gpus",
                        metavar="GPU",
                        help="Visible GPUs",
                        type=str,
                        nargs="+")
    parser.add_argument("--disable-mixed-precision",
                        dest="mixed_precision",
                        help="Disable mixed precision",
                        default=True,
                        action="store_false")
    parser.add_argument("--disable-xla",
                        dest="xla",
                        help="Disable XLA auto-clustering",
                        default=True,
                        action="store_false")
    parser.add_argument("--disable-profile",
                        dest="profile",
                        help="Disable profiling",
                        default=True,
                        action="store_false")
    return parser.parse_args()


def init(gpus: Union[List[str], None] = None,
         random_seed: Union[int, None] = None,
         mixed_precision: bool = True,
         xla: bool = True) -> tf.distribute.Strategy:
    """Init hardware and libraries."""
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        if mixed_precision:
            details = tf.config.experimental.get_device_details(device)
            compute_capability = details.get("compute_capability", None)
            if compute_capability and compute_capability[0] >= 7:
                keras.mixed_precision.set_global_policy("mixed_float16")
    if xla:
        tf.config.optimizer.set_jit("autoclustering")
    if random_seed is not None:
        keras.utils.set_random_seed(random_seed)
    strategy = tf.distribute.get_strategy()
    if len(physical_devices) > 1:
        strategy = tf.distribute.MirroredStrategy(
            devices=[device.name for device in physical_devices]
        )
    return strategy


def get_callbacks(
    output_dir: Union[str, None],
    checkpoint_dir: Union[str, None],
    log_dir: Union[str, None],
    monitor_metric: Union[str, None],
    play_ds: Union[tf.data.Dataset, None],
    early_stopping: int = 0,
    profile: bool = True
):
    """Get callbacks for training."""
    callbacks = []
    if output_dir is not None:
        if log_dir is None:
            log_dir = f"{output_dir}/logs"
        if checkpoint_dir is None:
            checkpoint_dir = f"{output_dir}/checkpoints"
    if log_dir is not None:
        callbacks.append(TensorBoard(
            log_dir=log_dir,
            histogram_freq=20,
            profile_batch=(5, 10) if profile else 0,
        ))
        if play_ds is not None:
            callbacks.append(PlayCallback(
                log_dir=f"{log_dir}/metrics",
                dataset=play_ds,
            ))
    if checkpoint_dir is not None:
        if monitor_metric is not None:
            callbacks.append(keras.callbacks.ModelCheckpoint(
                f"{checkpoint_dir}/best.weights.h5",
                monitor=monitor_metric,
                save_best_only=True,
                save_weights_only=True
            ))
        callbacks.append(keras.callbacks.ModelCheckpoint(
            f"{checkpoint_dir}/latest.weights.h5",
            save_best_only=False,
            save_weights_only=True
        ))
    if early_stopping > 0:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=early_stopping,
            verbose=1,
        ))
    return callbacks


def train(config: Dict[str, Any], strategy: tf.distribute.Strategy,
          profile: bool = True) -> None:
    """Run training."""
    LOG.info("Constructing models...")
    with strategy.scope():
        models = create_models(config["models"])
    if "train" in config:
        LOG.info("Loading datasets...")
        train_ds = create_train_dataset(**config["train_dataset"])
        if "val_dataset" in config:
            val_ds, play_ds = create_val_dataset(**config["val_dataset"])
            val_steps = (config["val_dataset"]["val_size"] //
                         config["val_dataset"]["batch_size"])
        else:
            val_ds = None
            play_ds = None
            val_steps = None
        train_model = models[config["train"]["model"]]
        LOG.info("Training %s", train_model.name)
        callbacks = get_callbacks(
            output_dir=config["train"].get("output_dir", None),
            log_dir=config["train"].get("log_dir", None),
            checkpoint_dir=config["train"].get("checkpoint_dir", None),
            monitor_metric=config["train"].get("monitor_metric", None),
            play_ds=play_ds,
            early_stopping=config["train"].get("early_stopping", 0),
            profile=profile,
        )
        if "unrolled_steps_per_execution" in config["train"]:
            train_model.unrolled_steps_per_execution = \
                config["train"]["unrolled_steps_per_execution"]
        train_model.fit(
            train_ds,
            validation_data=val_ds,
            validation_steps=val_steps,
            callbacks=callbacks,
            **{k: config["train"][k]
               for k in config["train"]
               if k not in ["model", "output_dir", "monitor_metric",
                            "early_stopping", "unrolled_steps_per_execution"]}
        )
    for model_name, export_config in config.get("export", {}).items():
        LOG.info("Exporting model %s", model_name)
        export_model = models[model_name]
        export_model.save_weights(export_config["weights_path"])
        with open(export_config["model_path"], "wt",
                  encoding="utf-8") as f:
            f.write(export_model.to_json())


def main(config_path: str, gpus: Union[List[str], None],
         mixed_precision: bool, xla: bool, profile: bool) -> int:
    """
    Run CLI.

    Parameters
    ----------
    config_path: str
        Path to config file
    gpus: Union[List[str], None]
        Visible GPUs
    mixed_precision: bool
        Use mixed precision for training if available
    xla: bool
        Use XLA autoclustering
    profile: bool
        Enable profiling

    Returns
    -------
    int
        Exit code
    """
    with open(config_path, "rt", encoding="utf-8") as f:
        config = yaml.unsafe_load(f)
    strategy = init(
        gpus=gpus,
        random_seed=config["seed"] if "seed" in config else None,
        mixed_precision=mixed_precision,
        xla=xla,
    )
    train(config, strategy, profile)
    return 0


if __name__ == "__main__":
    try:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(name)s: %(message)s",
            stream=sys.stderr,
        )
        sys.exit(main(**vars(parse_args())))
    except KeyboardInterrupt:
        sys.exit(-1)
    finally:
        logging.shutdown()
