# Flower Image Classification using ResNet50

This repository contains a PyTorch implementation of a ResNet 50 model fine-tuned on the 102 Category Flower Dataset. 

## Dataset
The project uses the [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), which consists of 8,189 images across 102 flower categories. 

## Model Architecture
The approach utilizes transfer learning with a pre-trained ResNet 50 model. The base weights were frozen to retain the model's feature extraction capabilities, and a custom fully connected layer with a Softmax activation was added to output probabilities for the 102 flower classes.

## Results
The model achieved a **92.43% Test Accuracy**.

Visualizations generated during evaluation:
* Loss & Accuracy Plot: `outputs/Res_plot.png`
* Predictions vs True Labels: `outputs/pred_vs_true_plot.png`

## Repository Structure
* `notebooks/`: Contains the original Jupyter Notebook (`Flowers102-Resnet50_v2.ipynb`).
* `src/`: Modularized Python scripts for data loading, model definition, training, and evaluation.
* `outputs/`: Saved visualizations from the training and evaluation phases.
