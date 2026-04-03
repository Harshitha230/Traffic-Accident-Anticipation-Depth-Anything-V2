import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import get_config
from utils import ensure_dir


def compute_grid_features(depth: np.ndarray, grid_size: int) -> np.ndarray:
    height, width = depth.shape
    row_edges = np.linspace(0, height, grid_size + 1, dtype=int)
    col_edges = np.linspace(0, width, grid_size + 1, dtype=int)

    features: list[float] = []
    for row_index in range(grid_size):
        for col_index in range(grid_size):
            patch = depth[row_edges[row_index] : row_edges[row_index + 1], col_edges[col_index] : col_edges[col_index + 1]]
            flat_patch = patch.reshape(-1)
            features.append(float(flat_patch.mean()))
            features.append(float(flat_patch.std()))
    return np.asarray(features, dtype=np.float32)


def compute_frame_features(depth_path: Path, bins: int, grid_size: int) -> np.ndarray:
    depth = np.asarray(Image.open(depth_path).convert("L"), dtype=np.float32) / 255.0
    flat = depth.reshape(-1)

    mean_depth = float(flat.mean())
    min_depth = float(flat.min())
    max_depth = float(flat.max())
    variance = float(flat.var())

    quantiles = np.quantile(flat, [0.25, 0.5, 0.75]).astype(np.float32)
    histogram, _ = np.histogram(flat, bins=bins, range=(0.0, 1.0), density=True)
    grid_features = compute_grid_features(depth, grid_size=grid_size)

    return np.concatenate(
        [
            np.array([mean_depth, min_depth, max_depth, variance], dtype=np.float32),
            quantiles.astype(np.float32),
            histogram.astype(np.float32),
            grid_features,
        ]
    )


def extract_sequence_features(depth_paths: list[Path], bins: int, grid_size: int) -> np.ndarray:
    features = [compute_frame_features(path, bins=bins, grid_size=grid_size) for path in depth_paths]
    return np.stack(features).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=4)
    args = parser.parse_args()

    config = get_config()
    ensure_dir(config.depth_features_dir)
    ensure_dir(config.processed_data_dir)

    depth_metadata = pd.read_csv(config.depth_metadata_path)
    source_df = pd.read_csv(args.csv_path) if args.csv_path else None

    if source_df is not None:
        sample_ids = source_df["sample_id"].astype(str).tolist()
        depth_metadata = depth_metadata[depth_metadata["sample_id"].astype(str).isin(sample_ids)].copy()
    elif args.split != "all":
        depth_metadata = depth_metadata[depth_metadata["split"] == args.split].copy()

    if args.limit is not None:
        depth_metadata = depth_metadata.head(args.limit).copy()

    feature_rows = []
    for _, row in tqdm(depth_metadata.iterrows(), total=len(depth_metadata), desc="Extracting depth features"):
        sample_id = row["sample_id"]
        depth_paths = [Path(path) for path in str(row["depth_paths"]).split("|") if path]
        output_path = config.depth_features_dir / f"{sample_id}.npy"

        if not output_path.exists() or args.overwrite:
            sequence_features = extract_sequence_features(depth_paths, bins=args.bins, grid_size=args.grid_size)
            np.save(output_path, sequence_features)
        else:
            sequence_features = np.load(output_path)

        feature_rows.append(
            {
                "sample_id": sample_id,
                "split": row["split"],
                "feature_path": str(output_path),
                "num_frames": int(sequence_features.shape[0]),
                "feature_dim": int(sequence_features.shape[1]),
            }
        )

    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(config.depth_features_metadata_path, index=False)
    print(f"Saved depth feature metadata to {config.depth_features_metadata_path}")
    if not feature_df.empty:
        print(feature_df[["num_frames", "feature_dim"]].describe())


if __name__ == "__main__":
    main()
