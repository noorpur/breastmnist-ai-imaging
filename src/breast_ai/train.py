from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainResult:
    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    best_val_loss: float
    stopped_epoch: int


def class_weights_from_loader(loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights from a labeled loader."""
    counts = torch.zeros(2, dtype=torch.float32)
    for _, y in loader:
        values, freqs = torch.unique(y, return_counts=True)
        for value, freq in zip(values, freqs):
            counts[int(value)] += float(freq)
    weights = counts.sum() / (2.0 * counts.clamp_min(1.0))
    return weights.to(device)


def _epoch_step(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device, train: bool) -> Dict[str, float]:
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(x.size(0))
    return {"loss": total_loss / total, "accuracy": correct / total}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
    min_delta: float = 1e-4,
    model_path: str | Path | None = None,
    use_class_weights: bool = True,
) -> TrainResult:
    """Train a model with class weighting, cosine scheduling, and early stopping."""
    model = model.to(device)
    class_weights = class_weights_from_loader(train_loader, device) if use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    rows: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_stats = _epoch_step(model, train_loader, criterion, optimizer, device, train=True)
        val_stats = _epoch_step(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "val_loss": val_stats["loss"],
            "val_accuracy": val_stats["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        rows.append(row)
        if val_stats["loss"] < best_val_loss - min_delta:
            best_val_loss = val_stats["loss"]
            best_epoch = epoch
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if model_path is not None:
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, model_path)
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history = pd.DataFrame(rows)
    return TrainResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=float(best_val_loss),
        stopped_epoch=int(history["epoch"].max()) if len(history) else 0,
    )


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Return true labels and malignant probabilities."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.extend(p.tolist())
        labels.extend(y.numpy().tolist())
    return np.asarray(labels, dtype=int), np.asarray(probs, dtype=float)


def loader_to_flat_arrays(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten normalized images for classical scikit-learn baselines."""
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.view(x.size(0), -1).numpy())
        ys.append(y.numpy())
    return np.vstack(xs), np.concatenate(ys).astype(int)
