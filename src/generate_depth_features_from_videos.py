import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

from config import get_config
from utils import ensure_dir, resolve_project_path


DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def choose_device() -> int:
    if torch.cuda.is_available():
        return 0
    return -1


def build_depth_pipeline(model_id: str):
    return pipeline("depth-estimation", model=model_id, device=choose_device())


def extract_video_frames(video_path: Path, expected_frames: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames_bgr = []
    while len(frames_bgr) < expected_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
    cap.release()

    if not frames_bgr:
        raise RuntimeError(f"No frames extracted from video: {video_path}")

    while len(frames_bgr) < expected_frames:
        frames_bgr.append(frames_bgr[-1].copy())

    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr[:expected_frames]]
    return [Image.fromarray(frame) for frame in frames_rgb]


def normalize_depth(depth_output) -> np.ndarray:
    if isinstance(depth_output, Image.Image):
        depth_array = np.asarray(depth_output, dtype=np.float32)
    elif hasattr(depth_output, "cpu"):
        depth_array = depth_output.cpu().numpy().astype(np.float32)
    else:
        depth_array = np.asarray(depth_output, dtype=np.float32)

    depth_array = np.squeeze(depth_array)
    depth_min = float(depth_array.min())
    depth_max = float(depth_array.max())
    if depth_max - depth_min < 1e-8:
        return np.zeros_like(depth_array, dtype=np.float32)
    return (depth_array - depth_min) / (depth_max - depth_min)


def compute_grid_features(depth: np.ndarray, grid_size: int) -> np.ndarray:
    height, width = depth.shape
    row_edges = np.linspace(0, height, grid_size + 1, dtype=int)
    col_edges = np.linspace(0, width, grid_size + 1, dtype=int)

    features = []
    for row_index in range(grid_size):
        for col_index in range(grid_size):
            patch = depth[row_edges[row_index]:row_edges[row_index + 1], col_edges[col_index]:col_edges[col_index + 1]]
            flat_patch = patch.reshape(-1)
            features.append(float(flat_patch.mean()))
            features.append(float(flat_patch.std()))
    return np.asarray(features, dtype=np.float32)


def compute_frame_features(depth: np.ndarray, bins: int, grid_size: int) -> np.ndarray:
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


def compute_sequence_features(depth_pipe, frames: list[Image.Image], bins: int, grid_size: int) -> np.ndarray:
    sequence_features = []
    for frame in frames:
        prediction = depth_pipe(frame)
        depth_image = prediction.get("depth")
        if depth_image is None:
            raise RuntimeError("Depth pipeline did not return a depth map.")
        normalized_depth = normalize_depth(depth_image)
        feature_vector = compute_frame_features(normalized_depth, bins=bins, grid_size=grid_size)
        sequence_features.append(feature_vector)
    return np.stack(sequence_features).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args()

    config = get_config()
    ensure_dir(config.depth_features_dir)
    ensure_dir(config.processed_data_dir)

    split_paths = {
        "train": config.train_split_path,
        "val": config.val_split_path,
        "test": config.test_split_path,
    }

    if args.csv_path is not None:
        split_df = pd.read_csv(args.csv_path).copy()
    elif args.split == "all":
        split_df = pd.concat([pd.read_csv(split_paths[name]) for name in ["train", "val", "test"]], ignore_index=True)
    else:
        split_df = pd.read_csv(split_paths[args.split]).copy()

    if args.limit is not None:
        split_df = split_df.head(args.limit).copy()

    depth_pipe = build_depth_pipeline(args.model_id)
    feature_rows = []

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Generating depth features from videos"):
        sample_id = row["sample_id"]
        split_name = row["split"]
        video_path = resolve_project_path(row["video_path"], config.project_root)
        sequence_length = int(row["sequence_length"])
        output_path = config.depth_features_dir / f"{sample_id}.npy"

        if output_path.exists() and not args.overwrite:
            sequence_features = np.load(output_path)
        else:
            frames = extract_video_frames(video_path, expected_frames=sequence_length)
            sequence_features = compute_sequence_features(
                depth_pipe,
                frames,
                bins=args.bins,
                grid_size=args.grid_size,
            )
            np.save(output_path, sequence_features)

        feature_rows.append(
            {
                "sample_id": sample_id,
                "split": split_name,
                "feature_path": str(output_path),
                "num_frames": int(sequence_features.shape[0]),
                "feature_dim": int(sequence_features.shape[1]),
            }
        )

    feature_df = pd.DataFrame(feature_rows)
    if config.depth_features_metadata_path.exists():
        existing_df = pd.read_csv(config.depth_features_metadata_path)
        feature_df = pd.concat([existing_df, feature_df], ignore_index=True)
        feature_df = feature_df.drop_duplicates(subset=["sample_id"], keep="last")
    feature_df = feature_df.sort_values(["split", "sample_id"]).reset_index(drop=True)
    feature_df.to_csv(config.depth_features_metadata_path, index=False)
    print(f"Saved depth feature metadata to {config.depth_features_metadata_path}")
    if not feature_df.empty:
        print(feature_df[["num_frames", "feature_dim"]].describe())


if __name__ == "__main__":
    main()
