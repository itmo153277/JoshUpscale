# Model Training

The training consists of two parts:

1. FRVSR training: train generator and flow net together directly using MSE.
2. GAN training: trains generator, flow net and discriminator. Generator is trained constantly, discriminator is trained depending on the balance ratio.

Model configuration schema can be found [here](./config-schema.json).

## Local training

Use `./train_local.py` for training in local environment. Refer to `./train_local.py --help` for more information..

## Google Colab

An example notebook for training on Google Colab with CPU/GPU/TPU is located [here](./train_colab.ipynb).

## Generate model for inference

Use `./optimize.py` for converting model weights into `model.pb` file, required for the GUI and AviSynth plugin. Refer to `./optimize.py --help` for more information.
