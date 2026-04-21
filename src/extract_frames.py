import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from config import get_config
from utils import ensure_dir, resolve_project_path


def extract_single_video(video_path: Path, output_dir: Path, expected_frames: int) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    while len(frames) < expected_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames extracted from video: {video_path}")

    while len(frames) < expected_frames:
        frames.append(frames[-1].copy())

    for frame_index, frame in enumerate(frames[:expected_frames]):
        output_path = output_dir / f"frame_{frame_index:03d}.jpg"
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Could not write frame to {output_path}")


def build_frame_metadata(split_df: pd.DataFrame, frames_root: Path) -> pd.DataFrame:
    rows = []
    for _, row in split_df.iterrows():
        sample_id = row["sample_id"]
        frame_dir = frames_root / sample_id
        frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
        rows.append(
            {
                "sample_id": sample_id,
                "frame_dir": str(frame_dir),
                "frame_paths": "|".join(str(path) for path in frame_paths),
                "num_extracted_frames": len(frame_paths),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = get_config()
    ensure_dir(config.frames_dir)
    ensure_dir(config.processed_data_dir)

    split_paths = {
        "train": config.train_split_path,
        "val": config.val_split_path,
        "test": config.test_split_path,
    }
    selected_splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    frames_metadata_rows = []

    split_dfs: list[tuple[str, pd.DataFrame]] = []
    if args.csv_path is not None:
        custom_df = pd.read_csv(args.csv_path)
        if args.limit is not None:
            custom_df = custom_df.head(args.limit).copy()
        split_name = custom_df["split"].iloc[0] if "split" in custom_df.columns and not custom_df.empty else "custom"
        split_dfs.append((split_name, custom_df))
    else:
        for split_name in selected_splits:
            split_df = pd.read_csv(split_paths[split_name])
            if args.limit is not None:
                split_df = split_df.head(args.limit).copy()
            split_dfs.append((split_name, split_df))

    for split_name, split_df in split_dfs:
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Extracting {split_name}"):
            sample_id = row["sample_id"]
            video_path = resolve_project_path(row["video_path"], config.project_root)
            output_dir = config.frames_dir / sample_id
            expected_frames = int(row["sequence_length"])

            existing_frames = sorted(output_dir.glob("frame_*.jpg"))
            if len(existing_frames) == expected_frames and not args.overwrite:
                continue

            ensure_dir(output_dir)
            if args.overwrite:
                for frame in existing_frames:
                    frame.unlink()

            extract_single_video(video_path, output_dir, expected_frames)

        frame_metadata = build_frame_metadata(split_df, config.frames_dir)
        frame_metadata["split"] = split_name
        frames_metadata_rows.append(frame_metadata)

    if frames_metadata_rows:
        combined_metadata = pd.concat(frames_metadata_rows, ignore_index=True)
        if config.extracted_frames_metadata_path.exists():
            existing_metadata = pd.read_csv(config.extracted_frames_metadata_path)
            combined_metadata = pd.concat([existing_metadata, combined_metadata], ignore_index=True)
            combined_metadata = combined_metadata.drop_duplicates(subset=["sample_id"], keep="last")
        combined_metadata = combined_metadata.sort_values(["split", "sample_id"]).reset_index(drop=True)
        combined_metadata.to_csv(config.extracted_frames_metadata_path, index=False)
        print(f"Saved frame metadata to {config.extracted_frames_metadata_path}")


if __name__ == "__main__":
    main()
