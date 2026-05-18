from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def find_last_conv(model: nn.Module) -> nn.Module:
    """Find the final convolutional layer for Grad-CAM."""
    convs = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not convs:
        raise ValueError("No Conv2d layer found for Grad-CAM.")
    return convs[-1]


def gradcam_heatmap(model: nn.Module, image: torch.Tensor, target_class: int = 1, target_layer: Optional[nn.Module] = None) -> np.ndarray:
    """Compute a Grad-CAM heatmap for a single image tensor shaped (1, H, W)."""
    model.eval()
    device = next(model.parameters()).device
    x = image.unsqueeze(0).to(device).requires_grad_(True)
    target_layer = target_layer or find_last_conv(model)
    activations = {}
    gradients = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output.detach().clone()

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0].detach().clone()

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    try:
        logits = model(x)
        score = logits[:, target_class].sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        acts = activations["value"]
        grads = gradients["value"]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1)).squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = torch.nn.functional.interpolate(cam[None, None], size=image.shape[-2:], mode="bilinear", align_corners=False)
        return cam.squeeze().detach().cpu().numpy()
    finally:
        handle_f.remove()
        handle_b.remove()


def save_gradcam_panel(model: nn.Module, loader, path: str | Path, max_items: int = 6, target_class: int = 1):
    """Save a panel of original images with Grad-CAM overlays."""
    images, labels = next(iter(loader))
    n = min(max_items, images.size(0))
    fig, axes = plt.subplots(2, n, figsize=(n * 1.8, 3.8))
    for i in range(n):
        img = images[i]
        heat = gradcam_heatmap(model, img, target_class=target_class)
        base = img.squeeze().numpy() * 0.5 + 0.5
        axes[0, i].imshow(base, cmap="gray")
        axes[0, i].set_title(f"label={int(labels[i])}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(base, cmap="gray")
        axes[1, i].imshow(heat, alpha=0.45)
        axes[1, i].axis("off")
    fig.suptitle("Grad-CAM overlays for malignant-class evidence", y=1.03)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
