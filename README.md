# Robust Anomaly Detection on CIFAR-10

Unsupervised anomaly detection system trained on clean CIFAR-10 images and evaluated against five types of synthetic corruptions. Five models are compared from a random baseline to convolutional autoencoders, with a full robustness sweep across corruption types and severity levels. An improved training phase demonstrates that with more data and longer training, the convolutional autoencoder surpasses classical methods.

## Results

### Final Model Comparison

| Model | Test AUROC |
|---|---|
| Improved CAE (latent_ch=64, 10k images, 10 epochs) | **0.9939** |
| Improved CAE (latent_ch=128) | 0.9920 |
| Improved CAE (latent_ch=32) | 0.9910 |
| PCA Reconstruction Baseline | 0.9772 |
| Isolation Forest | 0.9303 |
| Convolutional Autoencoder (initial) | 0.9139 |
| Denoising Autoencoder (initial) | 0.8281 |
| Random Baseline | 0.5141 |

### Improvement from Initial to Final

| Model | Initial AUROC | Improved AUROC | Gain |
|---|---|---|---|
| Convolutional Autoencoder | 0.914 | 0.994 | +0.080 |

Increasing training set from 4,000 to 10,000 images and epochs from 3 to 10 pushed the CAE from below PCA to above it — confirming that deep learning surpasses classical methods when given sufficient data.

### Robustness Sweep

The denoising autoencoder was evaluated across 5 corruption types at 3 severity levels:

| Corruption | Severity 1 | Severity 3 | Severity 5 |
|---|---|---|---|
| Gaussian Noise | 0.544 | 0.828 | 0.975 |
| Salt and Pepper | 0.659 | 0.877 | 0.958 |
| Brightness | 0.655 | 0.837 | 0.916 |
| Cutout | 0.622 | 0.803 | 0.952 |
| Gaussian Blur | 0.310 | 0.185 | 0.129 |

**Key finding:** Gaussian blur is the hardest corruption to detect. AUROC drops to 0.13 at severity 5 because blurred images remain structurally similar to clean images. Noise-based corruptions are detected most reliably at high severity.

## Models

### 1. Random Baseline
Assigns random anomaly scores. Confirms evaluation code is correct — expected AUROC near 0.50.

### 2. Isolation Forest
Trained on 15-dimensional image-statistic features (channel means, std, min, max, high-frequency energy). Learns the distribution of clean image statistics and scores anomalies as statistical outliers. Test AUROC: 0.930.

### 3. PCA Reconstruction Baseline
Fits PCA (64 components) on flattened clean images. Anomaly score is the reconstruction error after projecting to the low-dimensional space. Strong classical baseline at AUROC 0.977.

### 4. Convolutional Autoencoder (Initial)
3-layer encoder-decoder trained on 4,000 clean images for 3 epochs. Reconstruction error used as anomaly score. AUROC 0.914.

### 5. Denoising Autoencoder
Same architecture as CAE but trained to reconstruct clean images from corrupted inputs. Used for the robustness sweep. AUROC 0.828.

### 6. Improved Convolutional Autoencoder
Retrained with 10,000 images, 10 epochs, early stopping (patience=3), and latent dimension sweep (32, 64, 128 channels). Best result: latent_ch=64, AUROC 0.9939.

## Dataset

- **Source:** CIFAR-10 (torchvision)
- **Normal class:** Clean CIFAR-10 images
- **Anomaly class:** Programmatically corrupted images
- **Initial split:** 4,000 train / 1,000 validation / 1,000 test
- **Improved split:** 10,000 train / 1,000 validation / 1,000 test

## Corruption Types

- **Gaussian noise** — additive random noise
- **Salt and pepper** — random black and white pixels
- **Gaussian blur** — spatial smoothing
- **Brightness** — global intensity shift
- **Cutout** — random rectangular occlusion

Severity ranges from 1 (mild) to 5 (severe) across all corruption types.

## Tech Stack

Python, PyTorch, scikit-learn, NumPy, pandas, Matplotlib, tqdm

## Project Structure

```
anomaly-detection-cifar10/
├── initial_workflow_anomaly_detection.ipynb
├── README.md
├── results/
│   ├── convolutional_autoencoder_initial.pth
│   ├── denoising_autoencoder_initial.pth
│   ├── validation_model_comparison.csv
│   ├── final_test_model_comparison.csv
│   ├── robustness_sweep_initial.csv
│   ├── improved_model_comparison.csv
│   └── latent_dimension_sweep.csv
└── .gitignore
```

## How to Run

1. Install dependencies:
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm
```

2. Run all cells in `initial_workflow_anomaly_detection.ipynb`

3. CIFAR-10 downloads automatically via torchvision

## Key Takeaways

1. PCA reconstruction is a surprisingly strong baseline on small datasets
2. The convolutional autoencoder surpasses PCA when trained on 10,000+ images
3. Latent dimension of 64 is the sweet spot for this task
4. Gaussian blur is fundamentally harder to detect than other corruptions because it preserves image structure
5. Early stopping prevents overfitting and selects the best checkpoint automatically
