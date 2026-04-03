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
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve

from utils import ensure_dir, save_json


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_roc_curve(predictions_df: pd.DataFrame, output_path: Path, auc_value: float) -> None:
    fpr, tpr, _ = roc_curve(predictions_df["video_label"], predictions_df["predicted_probability"])
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_value:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Classifier ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion(predictions_df: pd.DataFrame, output_path: Path) -> None:
    cm = confusion_matrix(predictions_df["video_label"], predictions_df["predicted_label"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Crash"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Classifier Confusion Matrix")
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
        "--classifier-metrics-path",
        type=str,
        default="/Users/harshithamurali/Deep Learning/Final project/outputs/metrics/depth_video_logreg_200_results.json",
    )
    parser.add_argument(
        "--risk-history-path",
        type=str,
        default="/Users/harshithamurali/Deep Learning/Final project/ablation experiments/outputs/metrics/baseline_history.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/harshithamurali/Deep Learning/Final project/outputs/plots/classifier_baseline",
    )
    args = parser.parse_args()

    predictions_path = Path(args.predictions_path)
    classifier_metrics_path = Path(args.classifier_metrics_path)
    risk_history_path = Path(args.risk_history_path)
    output_dir = ensure_dir(Path(args.output_dir))

    predictions_df = pd.read_csv(predictions_path)
    classifier_metrics = load_json(classifier_metrics_path)
    if risk_history_path.exists():
        risk_history = load_json(risk_history_path)
        risk_summary = {
            "name": "Depth feature LSTM risk-score regressor",
            "best_validation_loss": float(np.min(risk_history["val_loss"])),
            "best_validation_mae": float(np.min(risk_history["val_mae"])),
            "best_validation_auc_roc": float(np.max(risk_history["val_auc_roc"])),
        }
    else:
        risk_summary = {
            "name": "Depth feature LSTM risk-score regressor",
            "best_validation_loss": None,
            "best_validation_mae": None,
            "best_validation_auc_roc": None,
        }

    roc_path = output_dir / "depth_video_classifier_roc.png"
    confusion_path = output_dir / "depth_video_classifier_confusion_matrix.png"
    plot_roc_curve(predictions_df, roc_path, classifier_metrics["val"]["auc_roc"])
    plot_confusion(predictions_df, confusion_path)

    summary = {
        "recommended_baseline": {
            "name": "Depth video-level logistic regression classifier",
            "validation_auc_roc": classifier_metrics["val"]["auc_roc"],
            "validation_accuracy": classifier_metrics["val"]["accuracy"],
            "validation_precision": classifier_metrics["val"]["precision"],
            "validation_recall": classifier_metrics["val"]["recall"],
        },
        "proposal_aligned_risk_model": risk_summary,
        "artifacts": {
            "roc_curve": str(roc_path),
            "confusion_matrix": str(confusion_path),
            "classifier_predictions": str(predictions_path),
            "classifier_metrics": str(classifier_metrics_path),
            "risk_history": str(risk_history_path) if risk_history_path.exists() else None,
        },
    }

    summary_path = output_dir / "classifier_result_summary.json"
    save_json(summary, summary_path)
    print(summary)


if __name__ == "__main__":
    main()
