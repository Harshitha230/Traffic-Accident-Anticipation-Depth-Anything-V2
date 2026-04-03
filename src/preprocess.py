import ast
import json
from pathlib import Path

import pandas as pd

from config import get_config
from utils import ensure_dir


def parse_crash_annotations(annotation_path: Path) -> dict[str, dict]:
    annotations: dict[str, dict] = {}
    with annotation_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            video_id = line.split(",[")[0]
            label_blob = line.split(",[")[1].split("],")[0]
            labels = ast.literal_eval("[" + label_blob + "]")
            start_frame, youtube_id, lighting, weather, ego_involve = line.split(",[")[1].split("],")[1].split(",")

            annotations[video_id] = {
                "frame_labels": labels,
                "start_frame": start_frame,
                "youtube_id": youtube_id,
                "lighting": lighting,
                "weather": weather,
                "ego_involve": ego_involve,
            }

    return annotations


def parse_split_entries(split_path: Path, split_name: str, dataset_root: Path, crash_annotations: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []

    with split_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            relative_feature_path, label_text = line.split()
            group, filename = relative_feature_path.split("/")
            video_id = Path(filename).stem
            is_crash = group == "positive"

            if is_crash:
                video_path = dataset_root / "Crash-1500" / f"{video_id}.mp4"
                annotation = crash_annotations.get(video_id)
                if annotation is None:
                    raise ValueError(f"Missing crash annotation for video {video_id}")
                frame_labels = annotation["frame_labels"]
                lighting = annotation["lighting"]
                weather = annotation["weather"]
                ego_involve = annotation["ego_involve"]
                start_frame = annotation["start_frame"]
                youtube_id = annotation["youtube_id"]
            else:
                video_path = dataset_root / "Normal" / f"{video_id}.mp4"
                frame_labels = [0] * 50
                lighting = "Unknown"
                weather = "Unknown"
                ego_involve = "Unknown"
                start_frame = ""
                youtube_id = ""

            rows.append(
                {
                    "sample_id": f"{group}_{video_id}",
                    "video_id": video_id,
                    "split": split_name,
                    "group": group,
                    "video_label": int(label_text),
                    "video_path": str(video_path),
                    "frame_labels": json.dumps(frame_labels),
                    "sequence_length": len(frame_labels),
                    "lighting": lighting,
                    "weather": weather,
                    "ego_involve": ego_involve,
                    "start_frame": start_frame,
                    "youtube_id": youtube_id,
                    "relative_feature_path": relative_feature_path,
                }
            )

    return rows


def build_metadata(dataset_root: Path, annotation_path: Path, train_split_path: Path, test_split_path: Path) -> pd.DataFrame:
    crash_annotations = parse_crash_annotations(annotation_path)
    rows = []
    rows.extend(parse_split_entries(train_split_path, "train", dataset_root, crash_annotations))
    rows.extend(parse_split_entries(test_split_path, "test", dataset_root, crash_annotations))

    metadata = pd.DataFrame(rows).sort_values(["split", "group", "video_id"]).reset_index(drop=True)
    return metadata


def main() -> None:
    config = get_config()
    ensure_dir(config.processed_data_dir)

    metadata = build_metadata(
        config.dataset_root,
        config.crash_annotation_path,
        config.official_train_split_path,
        config.official_test_split_path,
    )
    metadata.to_csv(config.metadata_path, index=False)
    print(f"Saved metadata with {len(metadata)} samples to {config.metadata_path}")
    print(metadata[["split", "video_label"]].value_counts().sort_index())


if __name__ == "__main__":
    main()
