import json
import os
import random
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_project_path(path: str | Path, project_root: str | Path) -> Path:
    candidate = Path(str(path))
    if candidate.exists():
        return candidate

    project_root = Path(project_root)
    raw = str(path)
    for anchor in ["CarCrashDataset", "data", "outputs", "checkpoints", "src", "scripts"]:
        token = f"/{anchor}/"
        if token in raw:
            suffix = raw.split(token, 1)[1]
            rewritten = project_root / anchor / suffix
            return rewritten

    return candidate


os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(Path("/tmp") / "matplotlib-cache")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2))


def save_checkpoint(state: dict, path: str | Path) -> None:
    if torch is None:
        raise ModuleNotFoundError("torch is required to save checkpoints")
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    if torch is None:
        raise ModuleNotFoundError("torch is required to load checkpoints")
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def plot_training_history(history: dict[str, list[float]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plt.figure(figsize=(10, 4))
    for key, values in history.items():
        plt.plot(values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
