# Object Detection with Pretrained YOLOv8

## Overview
This project implements an object detection model leveraging a pre-trained YOLOv8 architecture with a ResNet-50 backbone, customized using the PASCAL VOC 2007 dataset. The project demonstrates how to efficiently fine-tune a pre-trained model on a new dataset, while also incorporating hyperparameter tuning and custom data augmentations to enhance model performance.

This model is designed to detect a variety of object classes, such as aeroplanes, cars, animals, and more. The model has been modularized for flexibility and ease of use, with clear separation between data loading, model creation, and visualization.

## Key Features
- Utilizes the state-of-the-art YOLOv8 detector with a ResNet-50 backbone for high-performance object detection.
- Uses PASCAL VOC 2007 for training and evaluation with flexible preprocessing and augmentation.
- Includes configuration for batch size, learning rate, and number of training epochs, allowing for experimentation and tuning.
- Saves best weights to allow quick inference without retraining.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
By default, the model is set to train for 1 epoch for quick demonstration purposes. For best results, it is recommended to train for 10-35 epochs.

To train the model from scratch:
```bash
python src/main.py
```

### Configuration

You can modify the training configuration (e.g., number of epochs, learning rate) in the `src/config.py` file.

### Checkpoints

The program will save the best weights in the model_checkpoints/ directory.
