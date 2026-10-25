"""Figures: confusion matrices, training curves / summary, and misclassified examples."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix

from .config import DISPLAY_NAMES

CLASS_NAMES = ["Unsafe", "Safe"]


def plot_confusion_matrices(results: dict, output_path: Path):
    keys = list(results)
    ncols = 2 if len(keys) > 1 else 1
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    for ax, key in zip(axes.flat, keys):
        preds, labels = results[key]["preds"], results[key]["labels"]
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_pct = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True) * 100)
        annot = np.array([[f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)" for j in range(2)] for i in range(2)])
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 12})
        acc = (preds == labels).mean() * 100
        ax.set_title(f"{DISPLAY_NAMES.get(key, key)}\nAccuracy: {acc:.2f}%", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")

    plt.tight_layout()
    path = output_path / "confusion_matrices.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_training_summary(histories: dict, best_accs: dict, output_path: Path):
    """Training curves, validation loss and accuracy comparison for whatever experiments ran."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    markers = {"resnet50": "o", "vit": "s"}

    for mode, ax, title in (("rgb", axes[0], "RGB"), ("gray", axes[1], "Grayscale")):
        for key, hist in histories.items():
            if not key.endswith(mode):
                continue
            m = markers["vit" if key.startswith("vit") else "resnet50"]
            arch = "ViT" if key.startswith("vit") else "ResNet"
            ax.plot(hist["train_acc"], label=f"{arch} Train", linestyle="-", marker=m, markersize=3)
            ax.plot(hist["val_acc"], label=f"{arch} Val", linestyle="--", marker=m, markersize=3)
        ax.set(xlabel="Epoch", ylabel="Accuracy (%)", title=f"{title} preprocessing: training curves")
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend(loc="lower right")

    for key, hist in histories.items():
        axes[2].plot(hist["val_loss"], label=DISPLAY_NAMES.get(key, key), marker="o", markersize=3)
    axes[2].set(xlabel="Epoch", ylabel="Validation loss", title="Validation loss")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    path = output_path / "training_curves.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    names = [DISPLAY_NAMES.get(k, k).replace(" + ", "\n") for k in best_accs]
    accs = list(best_accs.values())
    bars = ax.bar(names, accs, color=["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"][: len(accs)],
                  alpha=0.8, edgecolor="black")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Model comparison")
    ax.set_ylim([max(0, min(accs) - 10), 100])
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{acc:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = output_path / "model_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_error_examples(indices, labels, preds, probs, image_ids, id_to_filename,
                        images_path, title, save_path: Path, n_max: int = 6):
    if len(indices) == 0:
        print(f"No examples for {title}")
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= min(n_max, len(indices)):
            continue
        idx = indices[i]
        filename = id_to_filename.get(image_ids[idx])
        if filename:
            try:
                ax.imshow(Image.open(os.path.join(images_path, filename)))
            except OSError:
                ax.text(0.5, 0.5, "Image not found", ha="center", va="center")
        true_label = CLASS_NAMES[labels[idx]]
        pred_label = CLASS_NAMES[preds[idx]]
        conf = probs[idx][preds[idx]] * 100
        ax.set_title(f"True: {true_label}, Pred: {pred_label}\nConf: {conf:.1f}%", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")
