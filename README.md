# Robust Anomaly Detection in Corrupted CIFAR-10

## Overview

This project investigates the robustness of anomaly detection models under corrupted image conditions. A convolutional autoencoder is trained on clean CIFAR-10 images and evaluated on synthetically corrupted data.

## Key Features

* Custom corruption pipeline (noise, blur, brightness, cutout)
* Reconstruction-based anomaly detection
* AUROC benchmarking across corruption types and severity levels
* Visualization of reconstruction outputs and anomaly scores

## Results

The model achieves strong anomaly detection performance (AUROC > 0.93) and demonstrates sensitivity to both corrupted and out-of-distribution inputs.

## Tech Stack

Python, PyTorch, NumPy, Matplotlib, Jupyter Notebook

## Key Insight

Anomaly detection models rely on learned data distributions, making them sensitive to both noise and unseen transformations such as rotation.
