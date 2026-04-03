import argparse
import json
import os
from pathlib import Path

matplotlib_cache = Path("/tmp") / "matplotlib-cache"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve

from utils import ensure_dir


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_metric_table(classifier_metrics: dict, risk_summary: dict, output_path: Path) -> None:
    rows = [
        {
            "Model": "Depth video classifier",
            "Task": "Video-level classification",
            "Subset": "200 videos",
            "AUC-ROC": round(classifier_metrics["validation_auc_roc"], 4),
            "Accuracy": round(classifier_metrics["validation_accuracy"], 4),
            "Precision": round(classifier_metrics["validation_precision"], 4),
            "Recall": round(classifier_metrics["validation_recall"], 4),
            "MAE": "-",
            "Val Loss": "-",
        },
        {
            "Model": "Depth LSTM risk regressor",
            "Task": "50-step risk-score regression",
            "Subset": "200 videos",
            "AUC-ROC": round(risk_summary["best_validation_auc_roc"], 4),
            "Accuracy": "-",
            "Precision": "-",
            "Recall": "-",
            "MAE": round(risk_summary["best_validation_mae"], 4),
            "Val Loss": round(risk_summary["best_validation_loss"], 4),
        },
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def plot_probability_histogram(predictions_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    normal_scores = predictions_df[predictions_df["video_label"] == 0]["predicted_probability"]
    crash_scores = predictions_df[predictions_df["video_label"] == 1]["predicted_probability"]
    plt.hist(normal_scores, bins=10, alpha=0.7, label="Normal", color="#4C78A8")
    plt.hist(crash_scores, bins=10, alpha=0.7, label="Crash", color="#E45756")
    plt.xlabel("Predicted Crash Probability")
    plt.ylabel("Count")
    plt.title("Predicted Probability Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_precision_recall(predictions_df: pd.DataFrame, output_path: Path) -> None:
    y_true = predictions_df["video_label"].to_numpy()
    scores = predictions_df["predicted_probability"].to_numpy()
    precision, recall, _ = precision_recall_curve(y_true, scores)
    display = PrecisionRecallDisplay(precision=precision, recall=recall)
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def choose_examples(predictions_df: pd.DataFrame) -> tuple[str, str]:
    normal_example = predictions_df[predictions_df["video_label"] == 0].sort_values("predicted_probability").iloc[0]["sample_id"]
    crash_example = predictions_df[predictions_df["video_label"] == 1].sort_values("predicted_probability", ascending=False).iloc[0]["sample_id"]
    return normal_example, crash_example


def find_frame_and_depth(sample_id: str, frame_metadata: pd.DataFrame, depth_metadata: pd.DataFrame, frame_index: int = 25) -> tuple[Path, Path]:
    frame_row = frame_metadata[frame_metadata["sample_id"] == sample_id].iloc[0]
    depth_row = depth_metadata[depth_metadata["sample_id"] == sample_id].iloc[0]
    frame_paths = [Path(path) for path in str(frame_row["frame_paths"]).split("|") if path]
    depth_paths = [Path(path) for path in str(depth_row["depth_paths"]).split("|") if path]
    return frame_paths[frame_index], depth_paths[frame_index]


def save_dataset_example_pair(sample_id: str, label_name: str, frame_metadata: pd.DataFrame, depth_metadata: pd.DataFrame, output_path: Path) -> None:
    rgb_path, depth_path = find_frame_and_depth(sample_id, frame_metadata, depth_metadata)
    rgb_image = Image.open(rgb_path).convert("RGB")
    depth_image = Image.open(depth_path).convert("L")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(rgb_image)
    axes[0].set_title(f"{label_name} RGB")
    axes[0].axis("off")
    axes[1].imshow(depth_image, cmap="inferno")
    axes[1].set_title(f"{label_name} Depth")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-path",
        type=str,
        default="/Users/harshithamurali/Deep Learning/Final project/outputs/predictions/depth_video_logreg_200_val_predictions.csv",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="/Users/harshithamurali/Deep Learning/Final project/outputs/plots/classifier_baseline/classifier_result_summary.json",
    )
    parser.add_argument(
        "--frame-metadata-path",
        type=str,
        default="/Users/harshithamurali/Deep Learning/Final project/data/processed/frame_metadata.csv",
    )
    parser.add_argument(
        "--depth-metadata-path",
        type=str,
        default="/Users/harshithamurali/Deep Learning/Final project/data/processed/depth_metadata.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/harshithamurali/Deep Learning/Final project/outputs/plots/presentation_artifacts",
    )
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))
    predictions_df = pd.read_csv(args.predictions_path)
    summary = load_json(Path(args.summary_path))
    frame_metadata = pd.read_csv(args.frame_metadata_path)
    depth_metadata = pd.read_csv(args.depth_metadata_path)

    save_metric_table(
        summary["recommended_baseline"],
        summary["proposal_aligned_risk_model"],
        output_dir / "model_comparison_table.csv",
    )
    plot_probability_histogram(predictions_df, output_dir / "predicted_probability_histogram.png")
    plot_precision_recall(predictions_df, output_dir / "precision_recall_curve.png")

    normal_example, crash_example = choose_examples(predictions_df)
    save_dataset_example_pair(
        normal_example,
        "Normal",
        frame_metadata,
        depth_metadata,
        output_dir / f"{normal_example}_dataset_example.png",
    )
    save_dataset_example_pair(
        crash_example,
        "Crash",
        frame_metadata,
        depth_metadata,
        output_dir / f"{crash_example}_dataset_example.png",
    )

    manifest = {
        "comparison_table": str(output_dir / "model_comparison_table.csv"),
        "probability_histogram": str(output_dir / "predicted_probability_histogram.png"),
        "precision_recall_curve": str(output_dir / "precision_recall_curve.png"),
        "normal_example": str(output_dir / f"{normal_example}_dataset_example.png"),
        "crash_example": str(output_dir / f"{crash_example}_dataset_example.png"),
        "normal_sample_id": normal_example,
        "crash_sample_id": crash_example,
    }
    (output_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(manifest)


if __name__ == "__main__":
    main()
