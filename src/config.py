from dataclasses import dataclass, field
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parents[1]
    dataset_root: Path = field(init=False)
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    depth_dir: Path = field(init=False)
    splits_dir: Path = field(init=False)
    frames_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)

    metadata_path: Path = field(init=False)
    train_split_path: Path = field(init=False)
    val_split_path: Path = field(init=False)
    test_split_path: Path = field(init=False)
    crash_annotation_path: Path = field(init=False)
    official_train_split_path: Path = field(init=False)
    official_test_split_path: Path = field(init=False)
    extracted_frames_metadata_path: Path = field(init=False)
    depth_metadata_path: Path = field(init=False)
    depth_features_dir: Path = field(init=False)
    depth_features_metadata_path: Path = field(init=False)
    rgb_features_dir: Path = field(init=False)
    rgb_features_metadata_path: Path = field(init=False)
    swift_module_cache_dir: Path = field(init=False)

    image_size: tuple[int, int] = (224, 224)
    sequence_length: int = 50
    batch_size: int = 8
    num_workers: int = 0
    num_classes: int = 2
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    seed: int = 42
    model_name: str = "baseline"
    checkpoint_name: str = "best_model.pt"

    depth_model_name: str = "midas_small"
    gaussian_sigma: float = 2.0
    device: str = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        self.dataset_root = self.project_root / "CarCrashDataset"
        self.data_dir = self.project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.depth_dir = self.data_dir / "depth"
        self.depth_features_dir = self.data_dir / "processed" / "depth_features"
        self.rgb_features_dir = self.data_dir / "processed" / "rgb_features"
        self.splits_dir = self.data_dir / "splits"
        self.frames_dir = self.data_dir / "raw" / "frames"
        self.outputs_dir = self.project_root / "outputs"
        self.checkpoints_dir = self.project_root / "checkpoints"

        self.metadata_path = self.processed_data_dir / "metadata.csv"
        self.train_split_path = self.splits_dir / "train.csv"
        self.val_split_path = self.splits_dir / "val.csv"
        self.test_split_path = self.splits_dir / "test.csv"
        self.crash_annotation_path = self.dataset_root / "Crash-1500.txt"
        self.official_train_split_path = self.dataset_root / "vgg16_features" / "train.txt"
        self.official_test_split_path = self.dataset_root / "vgg16_features" / "test.txt"
        self.extracted_frames_metadata_path = self.processed_data_dir / "frame_metadata.csv"
        self.depth_metadata_path = self.processed_data_dir / "depth_metadata.csv"
        self.depth_features_metadata_path = self.processed_data_dir / "depth_features_metadata.csv"
        self.rgb_features_metadata_path = self.processed_data_dir / "rgb_features_metadata.csv"
        self.swift_module_cache_dir = self.project_root.parent / ".swift-module-cache"


def get_config() -> Config:
    return Config()
