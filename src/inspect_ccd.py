import ast
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "CarCrashDataset"
SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "ccd_dataset_summary.json"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_crash_annotations(annotation_path: Path) -> dict:
    lighting_counts = Counter()
    weather_counts = Counter()
    ego_counts = Counter()
    malformed_rows = []
    accident_frames_per_video = []

    with annotation_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    for line_number, line in enumerate(lines, start=1):
        try:
            video_id = line.split(",[")[0]
            label_blob = line.split(",[")[1].split("],")[0]
            labels = ast.literal_eval("[" + label_blob + "]")
            start_frame, youtube_id, lighting, weather, ego_involve = line.split(",[")[1].split("],")[1].split(",")
        except Exception:
            malformed_rows.append(line_number)
            continue

        if len(labels) != 50:
            malformed_rows.append(line_number)
            continue

        accident_frames_per_video.append(sum(labels))
        lighting_counts[lighting] += 1
        weather_counts[weather] += 1
        ego_counts[ego_involve] += 1

        if not video_id.isdigit():
            malformed_rows.append(line_number)

    return {
        "rows": len(lines),
        "malformed_rows": malformed_rows,
        "lighting_counts": dict(lighting_counts),
        "weather_counts": dict(weather_counts),
        "ego_involvement_counts": dict(ego_counts),
        "accident_frames": {
            "min": min(accident_frames_per_video) if accident_frames_per_video else 0,
            "max": max(accident_frames_per_video) if accident_frames_per_video else 0,
            "avg": round(sum(accident_frames_per_video) / len(accident_frames_per_video), 2)
            if accident_frames_per_video
            else 0.0,
        },
    }


def parse_split_file(split_path: Path) -> dict:
    class_counts = Counter()
    missing_feature_entries = []
    missing_video_entries = []

    with split_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    for line in lines:
        relative_feature_path, label_text = line.split()
        class_counts[label_text] += 1

        feature_path = DATASET_ROOT / "vgg16_features" / relative_feature_path
        if not feature_path.exists():
            missing_feature_entries.append(relative_feature_path)

        split_folder, filename = relative_feature_path.split("/")
        video_id = Path(filename).stem
        if split_folder == "positive":
            video_path = DATASET_ROOT / "Crash-1500" / f"{video_id}.mp4"
        else:
            video_path = DATASET_ROOT / "Normal" / f"{video_id}.mp4"

        if not video_path.exists():
            missing_video_entries.append(str(video_path))

    return {
        "rows": len(lines),
        "class_counts": dict(class_counts),
        "missing_feature_entries": missing_feature_entries[:20],
        "missing_feature_count": len(missing_feature_entries),
        "missing_video_entries": missing_video_entries[:20],
        "missing_video_count": len(missing_video_entries),
    }


def count_videos(video_dir: Path) -> int:
    return len([path for path in video_dir.iterdir() if path.suffix.lower() == ".mp4"])


def main() -> None:
    crash_annotation_path = DATASET_ROOT / "Crash-1500.txt"
    train_split_path = DATASET_ROOT / "vgg16_features" / "train.txt"
    test_split_path = DATASET_ROOT / "vgg16_features" / "test.txt"

    summary = {
        "dataset_root": str(DATASET_ROOT),
        "exists": DATASET_ROOT.exists(),
        "crash_video_count": count_videos(DATASET_ROOT / "Crash-1500"),
        "normal_video_count": count_videos(DATASET_ROOT / "Normal"),
        "crash_annotation_exists": crash_annotation_path.exists(),
        "train_split_exists": train_split_path.exists(),
        "test_split_exists": test_split_path.exists(),
    }

    if crash_annotation_path.exists():
        summary["crash_annotations"] = parse_crash_annotations(crash_annotation_path)

    if train_split_path.exists():
        summary["train_split"] = parse_split_file(train_split_path)

    if test_split_path.exists():
        summary["test_split"] = parse_split_file(test_split_path)

    ensure_dir(SUMMARY_PATH.parent)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nSaved dataset summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
