from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, List


@dataclass
class ExperimentConfig:
    """Central configuration for the BreastMNIST research workflow.

    The default configuration favors research polish over speed: repeated CNN
    training across multiple random seeds, a high epoch budget, and early
    stopping. In practice the patience rule usually stops training before the
    maximum epoch budget is reached.
    """

    seeds: List[int] = field(default_factory=lambda: [13, 42, 101])
    seed: int = 42  # kept for backward-compatible single-seed utility calls
    batch_size: int = 64
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 30
    min_delta: float = 1e-4
    threshold: float = 0.5
    sensitivity_floor: float = 0.90
    num_workers: int = 0
    device: str = "auto"
    project_root: str = "."
    dataset_name: str = "breastmnist"
    image_size: int = 28
    use_class_weights: bool = True
    bootstrap_iterations: int = 1000

    def paths(self) -> Dict[str, Path]:
        root = Path(self.project_root).resolve()
        mapping = {
            "root": root,
            "data": root / "data",
            "figures": root / "figures",
            "results": root / "results",
            "analysis": root / "analysis",
            "models": root / "results" / "models",
            "tables": root / "results" / "tables",
        }
        for path in mapping.values():
            path.mkdir(parents=True, exist_ok=True)
        return mapping

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
