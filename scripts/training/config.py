# -*- coding: utf-8 -*-

"""Config definitions."""

import copy

DEFAULT_CONFIG = {
    "generator": {},
    "flow_model_type": "autoencoder",
    "flow": {},
    "full_model": {},
    "frvsr": {},
    "discriminator": {},
    "gan": {},
    "gan_train": {
        "generator_learning_rate": 0.0005,
        "flow_learning_rate": 0.0005,
        "discriminator_learning_rate": 0.0005,
        "steps_per_execution": 1,
        "compile_test_fn": False,
        "compile_play_fn": False,
    },
    "batch_size": 64
}

LARGE_MODEL = {
    "generator": {
        "num_blocks": 20,
        "num_filters": 32,
    },
    "flow_model_type": "autoencoder",
    "flow": {
        "filters": [32, 64, 128, 256, 128, 64, 32]
    }
}


def merge_configs(config1, config2):
    """
    Merge two configs.

    Parameters
    ----------
    config1 : dict
        Config 1
    config2 : dict
        Config 2

    Returns
    -------
    dict
        Merged config
    """
    config1 = copy.deepcopy(config1)
    keys = set(list(config1.keys()) + list(config2.keys()))
    for key in keys:
        if key not in config1:
            config1[key] = copy.deepcopy(config2[key])
        elif key in config2:
            if isinstance(config1[key], dict):
                config1[key] = merge_configs(config1[key], config2[key])
            else:
                config1[key] = copy.deepcopy(config2[key])
    return config1


def get_config(
    model=None,
    batch_size=None,
    crop_size=None,
    learning_rate=None,
    steps_per_execution=None,
    config_override=None
):
    """
    Create config for training.

    Parameters
    ----------
    model : str
        Model type
    batch_size : int
        Batch size
    crop_size : int
        Crop size
    learning_rate : float or dict
        Learning rate
    steps_per_execution : int or dict
        Steps per execution
    config_override : dict
        Overrides

    Returns
    -------
    dict
        Train config
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    if model == "large":
        config = merge_configs(config, LARGE_MODEL)
    elif model is not None:
        raise ValueError("unknown model")
    if batch_size is not None:
        config["batch_size"] = batch_size
    if crop_size is not None:
        config = merge_configs(config, {
            "frvsr": {
                "crop_size": crop_size
            },
            "gan": {
                "crop_size": crop_size
            },
            "discriminator": {
                "crop_size": crop_size
            }
        })
    if learning_rate is not None:
        if isinstance(learning_rate, dict):
            if "frvsr" in learning_rate:
                config["frvsr"]["learning_rate"] = learning_rate["frvsr"]
            if "gan_generator" in learning_rate:
                config["gan_train"]["generator_learning_rate"] = \
                    learning_rate["gan_generator"]
            if "gan_flow" in learning_rate:
                config["gan_train"]["flow_learning_rate"] = \
                    learning_rate["gan_flow"]
            if "gan_discriminator" in learning_rate:
                config["gan_train"]["discriminator_learning_rate"] = \
                    learning_rate["gan_discriminator"]
        else:
            config = merge_configs(config, {
                "frvsr": {
                    "learning_rate": learning_rate
                },
                "gan_train": {
                    "generator_learning_rate": learning_rate,
                    "flow_learning_rate": learning_rate,
                    "discriminator_learning_rate": learning_rate,
                },
            })
    if steps_per_execution is not None:
        if isinstance(steps_per_execution, dict):
            if "frvsr" in steps_per_execution:
                config["frvsr"]["steps_per_execution"] = \
                    steps_per_execution["frvsr"]
            if "gan" in steps_per_execution:
                config["gan_train"]["steps_per_execution"] = \
                    steps_per_execution["gan"]
        else:
            config = merge_configs(config, {
                "frvsr": {
                    "steps_per_execution": steps_per_execution
                },
                "gan_train": {
                    "steps_per_execution": steps_per_execution
                }
            })
    if config_override is not None:
        config = merge_configs(config, config_override)
    return config
