# Flower Species Classification with PyTorch

Transfer learning for **102 flower species**, developed as part of Udacity's *AI Programming with Python* Nanodegree.

## Overview

This project trains a neural network to classify flower photographs into 102 categories. It uses a pretrained **VGG16** or **DenseNet121** backbone with frozen feature-extraction layers and a trainable classifier head. It includes command-line scripts for training and top-k inference, plus the original exploratory notebook.

## Features

- Image augmentation for training; deterministic preprocessing for validation and testing
- Transfer learning with torchvision pretrained models
- Validation loss and accuracy after each epoch
- Held-out test evaluation, JSON metrics and portable state-dictionary checkpoints
- Top-k predictions mapped to human-readable flower names
- CPU, CUDA and Apple Silicon MPS device support (subject to local PyTorch compatibility)

## Repository contents

- `train.py`: train, validate, evaluate and save a model
- `predict.py`: load a checkpoint and predict image classes
- `Image Classifier Project.ipynb`: original project notebook
- `cat_to_name.json`: category-to-name mapping
- `tests/`: lightweight unit tests

## Dataset

The Udacity flower dataset contains 102 classes, with separate `train/`, `valid/`, and `test/` directories. The image data is **not included** in this repository. Obtain the dataset through your authorised course materials and arrange it as:

```text
flowers/
  train/<class_id>/*.jpg
  valid/<class_id>/*.jpg
  test/<class_id>/*.jpg
```

## Installation

Use Python 3.10+ and a compatible installation of PyTorch and torchvision. Install PyTorch using the [official installation guide](https://pytorch.org/get-started/locally/) for your operating system, then:

```bash
python -m pip install -r requirements.txt
```

## Training

```bash
python train.py flowers --arch vgg16 --epochs 5 --hidden_units 512 --learning_rate 0.001 --save_dir checkpoints/flowers.pth
```

For CUDA-capable systems, add `--gpu`. On supported Apple Silicon Macs, MPS is selected automatically when CUDA is not requested and available.

Training writes a checkpoint and a companion `.metrics.json` file. The test set is evaluated after training, not used for parameter updates.

## Prediction

```bash
python predict.py flowers/test/58/example.jpg checkpoints/flowers.pth --top_k 5 --category_names cat_to_name.json
```

Replace the example image path with an actual image from your dataset.

**Important:** The refactored prediction script expects a checkpoint produced by the refactored training script. Checkpoints from the original notebook/scripts may have a different format and should be regenerated.

## Testing

```bash
python -m pytest -q
```

## Results

No new performance numbers are claimed here until the refactored scripts are run on the flower dataset. After running training, use the generated metrics file to report actual held-out accuracy and loss. Results can vary with the architecture, seed, training duration and hardware.

## Limitations and next steps

- Classification is restricted to the 102 trained flower categories.
- Performance on out-of-distribution images has not been established.
- Future work could add confidence calibration, class-level evaluation and a lightweight demo.

## Attribution

Built for Udacity's *AI Programming with Python* Nanodegree using PyTorch and torchvision. Original notebook and coursework context are retained for transparency. Refer to the repository's LICENSE for code licensing; dataset and pretrained model assets may have separate terms.
