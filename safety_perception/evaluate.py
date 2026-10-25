"""Inference helpers for error analysis."""

import numpy as np
import torch
from tqdm import tqdm


def get_predictions(model, loader, device):
    """Return predictions, labels, image ids and softmax probabilities."""
    model.eval()
    preds, labels, image_ids, probs = [], [], [], []
    with torch.no_grad():
        for imgs, batch_labels, batch_ids in tqdm(loader, desc="Getting predictions"):
            outputs = model(imgs.to(device))
            batch_probs = torch.softmax(outputs, dim=1)
            preds.extend(outputs.argmax(1).cpu().numpy())
            labels.extend(batch_labels.numpy())
            image_ids.extend(batch_ids)
            probs.extend(batch_probs.cpu().numpy())
    return np.array(preds), np.array(labels), list(image_ids), np.array(probs)


def split_errors(preds, labels):
    """False positives (unsafe predicted safe) and false negatives (safe predicted unsafe)."""
    errors = np.where(preds != labels)[0]
    false_pos = [i for i in errors if labels[i] == 0 and preds[i] == 1]
    false_neg = [i for i in errors if labels[i] == 1 and preds[i] == 0]
    return errors, false_pos, false_neg
