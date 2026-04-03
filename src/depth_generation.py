import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

from config import get_config
from utils import ensure_dir


DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def choose_device() -> int:
    if torch.cuda.is_available():
        return 0
    return -1


def build_depth_pipeline(model_id: str):
    return pipeline("depth-estimation", model=model_id, device=choose_device())


def normalize_depth_to_uint8(depth_output) -> np.ndarray:
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
        return np.zeros_like(depth_array, dtype=np.uint8)
    normalized = (depth_array - depth_min) / (depth_max - depth_min)
    return (normalized * 255).astype(np.uint8)


def save_depth_map(depth_pipe, frame_path: Path, output_path: Path) -> None:
    image = Image.open(frame_path).convert("RGB")
    prediction = depth_pipe(image)
    depth_image = prediction.get("depth")
    if depth_image is None:
        raise RuntimeError(f"No depth output returned for {frame_path}")
    depth_array = normalize_depth_to_uint8(depth_image)
    Image.fromarray(depth_array).save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args()

    config = get_config()
    ensure_dir(config.depth_dir)
    ensure_dir(config.processed_data_dir)

    frame_metadata = pd.read_csv(config.extracted_frames_metadata_path)
    if args.csv_path is not None:
        target_samples = pd.read_csv(args.csv_path)["sample_id"].astype(str).tolist()
        frame_metadata = frame_metadata[frame_metadata["sample_id"].astype(str).isin(target_samples)].copy()
    elif args.split != "all":
        frame_metadata = frame_metadata[frame_metadata["split"] == args.split].copy()
    if args.limit is not None:
        frame_metadata = frame_metadata.head(args.limit).copy()

    depth_pipe = build_depth_pipeline(args.model_id)
    depth_rows = []

    for _, row in tqdm(frame_metadata.iterrows(), total=len(frame_metadata), desc="Generating depth maps"):
        sample_id = row["sample_id"]
        frame_paths = [Path(path) for path in str(row["frame_paths"]).split("|") if path]
        sample_depth_dir = ensure_dir(config.depth_dir / sample_id)

        existing_depths = sorted(sample_depth_dir.glob("frame_*.png"))
        if args.overwrite:
            for path in existing_depths:
                path.unlink()
            existing_depths = []

        if len(existing_depths) != len(frame_paths):
            for frame_path in frame_paths:
                output_path = sample_depth_dir / f"{frame_path.stem}.png"
                if output_path.exists() and not args.overwrite:
                    continue
                save_depth_map(depth_pipe, frame_path, output_path)

        saved_depth_paths = sorted(sample_depth_dir.glob("frame_*.png"))
        depth_rows.append(
            {
                "sample_id": sample_id,
                "split": row["split"],
                "depth_dir": str(sample_depth_dir),
                "depth_paths": "|".join(str(path) for path in saved_depth_paths),
                "num_depth_frames": len(saved_depth_paths),
            }
        )

    pd.DataFrame(depth_rows).to_csv(config.depth_metadata_path, index=False)
    print(f"Saved depth metadata to {config.depth_metadata_path}")


if __name__ == "__main__":
    main()
