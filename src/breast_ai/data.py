from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: Tuple[str, str]
    dataset_sizes: Dict[str, int]
    positive_label_name: str = "malignant"


class ArrayDataset(Dataset):
    """Small image dataset backed by numpy arrays.

    The MedMNIST arrays arrive as HxW grayscale images with labels in a shape
    such as (N, 1). This class normalizes each image to roughly [-1, 1] and
    returns tensors ready for PyTorch models.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.images = images.astype("float32") / 255.0
        self.labels = labels.reshape(-1).astype("int64")
        self.augment = augment

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, idx: int):
        image = self.images[idx]
        if image.ndim == 2:
            image = image[None, :, :]
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = image.transpose(2, 0, 1)
        x = torch.tensor(image, dtype=torch.float32)
        if self.augment:
            # Lightweight augmentations that do not require torchvision.
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[2])
            if torch.rand(()) < 0.25:
                x = torch.clamp(x + 0.03 * torch.randn_like(x), 0.0, 1.0)
        x = (x - 0.5) / 0.5
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def load_breastmnist_arrays(root: str = "data", download: bool = True):
    """Load BreastMNIST arrays.

    A local ``breastmnist.npz`` file is read directly when present. This avoids
    brittle torchvision download paths and makes Google Drive reruns more
    reliable. If the file is absent, the official MedMNIST package is used to
    download it.
    """
    from pathlib import Path

    root_path = Path(root)
    npz_path = root_path / "breastmnist.npz"
    if npz_path.exists():
        arrays = np.load(npz_path)
        return (
            arrays["train_images"], arrays["train_labels"],
            arrays["val_images"], arrays["val_labels"],
            arrays["test_images"], arrays["test_labels"],
        )

    try:
        import medmnist
        from medmnist import INFO
    except Exception as exc:  # pragma: no cover - friendly runtime message
        raise ImportError(
            "medmnist is required. In Colab, run the setup cell or `pip install medmnist`."
        ) from exc

    info = INFO["breastmnist"]
    dataset_cls = getattr(medmnist, info["python_class"])
    train = dataset_cls(split="train", root=root, download=download, size=28)
    val = dataset_cls(split="val", root=root, download=download, size=28)
    test = dataset_cls(split="test", root=root, download=download, size=28)
    return train.imgs, train.labels, val.imgs, val.labels, test.imgs, test.labels


def make_dataloaders(
    root: str = "data",
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    download: bool = True,
) -> DatasetBundle:
    """Create PyTorch dataloaders for BreastMNIST."""
    train_x, train_y, val_x, val_y, test_x, test_y = load_breastmnist_arrays(root, download)
    generator = torch.Generator().manual_seed(seed)
    train_ds = ArrayDataset(train_x, train_y, augment=True)
    val_ds = ArrayDataset(val_x, val_y, augment=False)
    test_ds = ArrayDataset(test_x, test_y, augment=False)

    return DatasetBundle(
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator),
        val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        class_names=("non_malignant", "malignant"),
        dataset_sizes={"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
    )


def summarize_label_distribution(loader: DataLoader, class_names: Tuple[str, str]) -> Dict[str, int]:
    """Count labels in a dataloader."""
    counts = {name: 0 for name in class_names}
    for _, y in loader:
        values, freqs = torch.unique(y, return_counts=True)
        for value, freq in zip(values.tolist(), freqs.tolist()):
            counts[class_names[int(value)]] += int(freq)
    return counts
