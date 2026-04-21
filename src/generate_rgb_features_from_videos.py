import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import models

from config import get_config
from utils import ensure_dir, resolve_project_path


DEFAULT_MODEL_NAME = "resnet18"


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_rgb_encoder(model_name: str, device: torch.device):
    if model_name != DEFAULT_MODEL_NAME:
        raise ValueError(f"Unsupported RGB model: {model_name}. Supported models: {DEFAULT_MODEL_NAME}")

    weights = models.ResNet18_Weights.DEFAULT
    backbone = models.resnet18(weights=weights)
    encoder = torch.nn.Sequential(*list(backbone.children())[:-1]).to(device)
    encoder.eval()
    return encoder, weights.transforms()


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


def compute_sequence_features(
    encoder: torch.nn.Module,
    preprocess,
    frames: list[Image.Image],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    sequence_features = []
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            frame_batch = frames[start:start + batch_size]
            batch_tensor = torch.stack([preprocess(frame) for frame in frame_batch]).to(device)
            encoded = encoder(batch_tensor).flatten(1).cpu().numpy().astype(np.float32)
            sequence_features.append(encoded)
    return np.concatenate(sequence_features, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()

    config = get_config()
    ensure_dir(config.rgb_features_dir)
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

    device = choose_device()
    encoder, preprocess = build_rgb_encoder(args.model_name, device)
    feature_rows = []

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Generating RGB features from videos"):
        sample_id = row["sample_id"]
        split_name = row["split"]
        video_path = resolve_project_path(row["video_path"], config.project_root)
        sequence_length = int(row["sequence_length"])
        output_path = config.rgb_features_dir / f"{sample_id}.npy"

        if output_path.exists() and not args.overwrite:
            sequence_features = np.load(output_path)
        else:
            frames = extract_video_frames(video_path, expected_frames=sequence_length)
            sequence_features = compute_sequence_features(
                encoder,
                preprocess,
                frames,
                batch_size=args.batch_size,
                device=device,
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
    if config.rgb_features_metadata_path.exists():
        existing_df = pd.read_csv(config.rgb_features_metadata_path)
        feature_df = pd.concat([existing_df, feature_df], ignore_index=True)
        feature_df = feature_df.drop_duplicates(subset=["sample_id"], keep="last")
    feature_df = feature_df.sort_values(["split", "sample_id"]).reset_index(drop=True)
    feature_df.to_csv(config.rgb_features_metadata_path, index=False)
    print(f"Saved RGB feature metadata to {config.rgb_features_metadata_path}")
    if not feature_df.empty:
        print(feature_df[["num_frames", "feature_dim"]].describe())


if __name__ == "__main__":
    main()
