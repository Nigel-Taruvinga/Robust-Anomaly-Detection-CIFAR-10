import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score


def reconstruction_scores(model, loader, device="cpu"):
    """
    Returns reconstruction error per image (MSE per sample)
    """
    model.eval()
    model.to(device)

    criterion = nn.MSELoss(reduction="none")

    scores = []

    with torch.no_grad():
        for clean, corrupted, _ in loader:
            clean = clean.to(device)
            corrupted = corrupted.to(device)

            recon = model(corrupted)

            # per-sample reconstruction error
            loss_map = criterion(recon, corrupted)  # shape: [B, C, H, W]
            per_sample = loss_map.mean(dim=(1, 2, 3)).detach().cpu().numpy()

            scores.extend(per_sample.tolist())

    return np.array(scores)


def compute_auroc(clean_scores, anomaly_scores):
    """
    clean_scores: reconstruction errors for clean samples
    anomaly_scores: reconstruction errors for anomalies/corrupted samples
    """
    y_true = np.concatenate([
        np.zeros(len(clean_scores)),
        np.ones(len(anomaly_scores))
    ])
    y_score = np.concatenate([clean_scores, anomaly_scores])

    return roc_auc_score(y_true, y_score)
