import ast
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config, get_config
from utils import resolve_project_path


FeatureMode = Literal["depth", "rgb", "rgb_depth"]


def parse_sequence(text: str) -> list[float]:
    return list(ast.literal_eval(text))


def _rename_feature_columns(feature_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return feature_df.rename(
        columns={
            "feature_path": f"{prefix}_feature_path",
            "num_frames": f"{prefix}_num_frames",
            "feature_dim": f"{prefix}_feature_dim",
        }
    )


def _load_feature_metadata(path: str | Path) -> pd.DataFrame:
    feature_df = pd.read_csv(path).copy()
    feature_df["sample_id"] = feature_df["sample_id"].astype(str)
    feature_df["split"] = feature_df["split"].astype(str)
    return feature_df


class RiskSequenceDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        config: Config,
        feature_mode: FeatureMode = "depth",
        include_depth_delta: bool = False,
    ) -> None:
        self.config = config
        self.feature_mode = feature_mode
        self.include_depth_delta = include_depth_delta

        split_df = pd.read_csv(csv_path).copy()
        split_df["sample_id"] = split_df["sample_id"].astype(str)
        split_df["split"] = split_df["split"].astype(str)

        merged = self._merge_features(split_df)
        if merged.empty:
            raise ValueError(f"No matching feature rows found for {csv_path}")

        self.data = merged.reset_index(drop=True)

    def _merge_features(self, split_df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_mode == "depth":
            depth_df = _rename_feature_columns(_load_feature_metadata(self.config.depth_features_metadata_path), "depth")
            return split_df.merge(depth_df, on=["sample_id", "split"], how="inner")

        if self.feature_mode == "rgb":
            rgb_df = _rename_feature_columns(_load_feature_metadata(self.config.rgb_features_metadata_path), "rgb")
            return split_df.merge(rgb_df, on=["sample_id", "split"], how="inner")

        if self.feature_mode == "rgb_depth":
            depth_df = _rename_feature_columns(_load_feature_metadata(self.config.depth_features_metadata_path), "depth")
            rgb_df = _rename_feature_columns(_load_feature_metadata(self.config.rgb_features_metadata_path), "rgb")
            merged = split_df.merge(depth_df, on=["sample_id", "split"], how="inner")
            merged = merged.merge(rgb_df, on=["sample_id", "split"], how="inner")
            return merged

        raise ValueError(f"Unsupported feature mode: {self.feature_mode}")

    def __len__(self) -> int:
        return len(self.data)

    def _load_feature_array(self, row: pd.Series, prefix: str) -> np.ndarray:
        feature_path = resolve_project_path(row[f"{prefix}_feature_path"], self.config.project_root)
        return np.load(feature_path).astype(np.float32)

    def _append_depth_delta(self, depth_array: np.ndarray) -> np.ndarray:
        delta = np.zeros_like(depth_array, dtype=np.float32)
        delta[1:] = depth_array[1:] - depth_array[:-1]
        return np.concatenate([depth_array, delta], axis=-1)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        row = self.data.iloc[index]

        if self.feature_mode == "depth":
            feature_array = self._load_feature_array(row, "depth")
            if self.include_depth_delta:
                feature_array = self._append_depth_delta(feature_array)
        elif self.feature_mode == "rgb":
            feature_array = self._load_feature_array(row, "rgb")
        else:
            depth_array = self._load_feature_array(row, "depth")
            rgb_array = self._load_feature_array(row, "rgb")
            if depth_array.shape[0] != rgb_array.shape[0]:
                raise ValueError(
                    f"Mismatched sequence lengths for sample {row['sample_id']}: "
                    f"depth={depth_array.shape[0]}, rgb={rgb_array.shape[0]}"
                )
            if self.include_depth_delta:
                depth_array = self._append_depth_delta(depth_array)
            feature_array = np.concatenate([rgb_array, depth_array], axis=-1)

        risk_curve = np.asarray(parse_sequence(row["risk_curve"]), dtype=np.float32)
        frame_labels = np.asarray(parse_sequence(row["frame_labels"]), dtype=np.float32)

        return {
            "features": torch.from_numpy(feature_array),
            "risk_curve": torch.from_numpy(risk_curve),
            "frame_labels": torch.from_numpy(frame_labels),
            "video_label": torch.tensor(int(row["video_label"]), dtype=torch.long),
            "accident_onset_index": torch.tensor(int(row["accident_onset_index"]), dtype=torch.long),
            "sample_id": row["sample_id"],
            "video_id": str(row["video_id"]).zfill(6),
        }


def get_sequence_dataloaders(
    config: Config | None = None,
    train_csv_path: str | Path | None = None,
    val_csv_path: str | Path | None = None,
    test_csv_path: str | Path | None = None,
    feature_mode: FeatureMode = "depth",
    include_depth_delta: bool = False,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    config = config or get_config()
    train_csv_path = Path(train_csv_path or config.train_split_path)
    val_csv_path = Path(val_csv_path or config.val_split_path)

    train_dataset = RiskSequenceDataset(
        train_csv_path,
        config,
        feature_mode=feature_mode,
        include_depth_delta=include_depth_delta,
    )
    val_dataset = RiskSequenceDataset(
        val_csv_path,
        config,
        feature_mode=feature_mode,
        include_depth_delta=include_depth_delta,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    test_loader = None
    if test_csv_path is not None:
        test_dataset = RiskSequenceDataset(
            Path(test_csv_path),
            config,
            feature_mode=feature_mode,
            include_depth_delta=include_depth_delta,
        )
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    return train_loader, val_loader, test_loader
