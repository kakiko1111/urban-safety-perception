"""Training loop with per-epoch validation and best-checkpoint saving."""

import copy
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm


def _run_epoch(model, loader, criterion, device, optimizer=None, desc=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    iterator = tqdm(loader, desc=desc) if training else loader
    with torch.set_grad_enabled(training):
        for imgs, labels in iterator:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
            if training:
                iterator.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100.0 * correct / total:.1f}%")

    return total_loss / len(loader), 100.0 * correct / total


def train_model(model, train_loader, val_loader, model_name, cfg, device, output_path: Path):
    """AdamW + cosine annealing. Returns the best-validation model, history and best accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state = -1.0, None  # -1 so epoch 1 is always saved

    print(f"\n{'=' * 60}\nTraining {model_name}\n{'=' * 60}")
    for epoch in range(cfg["num_epochs"]):
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, device, optimizer, desc=f"Epoch {epoch + 1}/{cfg['num_epochs']}"
        )
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)
        print(f"Epoch {epoch + 1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # deepcopy: state_dict() tensors share memory with the live parameters,
            # so a shallow copy would silently follow later updates.
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output_path / f"{model_name}_best.pth")
            print(f"  -> New best! Saved to {model_name}_best.pth")

    model.load_state_dict(best_state)
    return model, history, best_val_acc
