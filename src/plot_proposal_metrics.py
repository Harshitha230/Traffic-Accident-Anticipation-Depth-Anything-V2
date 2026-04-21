import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def extract_window_metrics(analysis: dict) -> tuple[list[str], list[float], list[float], list[float]]:
    keys = [
        ("pre_onset_min_gap_0", "Pre-onset"),
        ("pre_onset_min_gap_5", ">=5 frames"),
        ("pre_onset_min_gap_10", ">=10 frames"),
    ]
    labels, rgb_vals, fused_vals, gains = [], [], [], []
    for key, label in keys:
        entry = analysis[key]["all"]
        labels.append(label)
        rgb_vals.append(float(entry["rgb"]["auc_roc_pre_onset"]))
        fused_vals.append(float(entry["rgb_depth"]["auc_roc_pre_onset"]))
        gains.append(float(entry["auc_gain_rgb_depth_minus_rgb"]))
    return labels, rgb_vals, fused_vals, gains


def plot_window_comparison(labels, rgb_vals, fused_vals, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars1 = ax.bar(x - width / 2, rgb_vals, width, label="RGB-only", color="#4C78A8")
    bars2 = ax.bar(x + width / 2, fused_vals, width, label="RGB+Depth", color="#E45756")

    ax.set_title("Early Anticipation AUC Comparison")
    ax.set_ylabel("AUC-ROC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.55, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.008, f"{height:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_gain(labels, gains, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(x, gains, width=0.5, color="#54A24B")
    ax.set_title("Depth Contribution to Early Anticipation")
    ax.set_ylabel("AUC Gain over RGB-only")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, max(gains) * 1.3)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.006, f"+{height:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_proposal_summary(rgb_metrics: dict, fused_metrics: dict, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    labels = ["Pre-onset AUC", "Warning recall @0.5", "Avg lead (frames)"]
    rgb_vals = [
        float(rgb_metrics["auc_roc_pre_onset"]),
        float(rgb_metrics["tta_threshold_0.50_warning_recall"]),
        float(rgb_metrics["tta_threshold_0.50_frames"]),
    ]
    fused_vals = [
        float(fused_metrics["auc_roc_pre_onset"]),
        float(fused_metrics["tta_threshold_0.50_warning_recall"]),
        float(fused_metrics["tta_threshold_0.50_frames"]),
    ]

    x = np.arange(len(labels))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), gridspec_kw={"width_ratios": [2.1, 1.0]})

    bars1 = axes[0].bar(x[:2] - width / 2, rgb_vals[:2], width, label="RGB-only", color="#4C78A8")
    bars2 = axes[0].bar(x[:2] + width / 2, fused_vals[:2], width, label="RGB+Depth", color="#E45756")
    axes[0].set_title("Proposal-Aligned Metrics")
    axes[0].set_ylabel("Score")
    axes[0].set_xticks(x[:2])
    axes[0].set_xticklabels(labels[:2])
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].legend(frameon=False, loc="lower right")
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, height + 0.015, f"{height:.3f}", ha="center", va="bottom", fontsize=9)

    lead_bars = axes[1].bar([0, 1], [rgb_vals[2], fused_vals[2]], color=["#4C78A8", "#E45756"], width=0.5)
    axes[1].set_title("Avg Warning Lead")
    axes[1].set_ylabel("Frames")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["RGB-only", "RGB+Depth"])
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    for bar in lead_bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, height + 0.05, f"{height:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-json", default="outputs/metrics/rgb_vs_rgbdepth_anticipation_subset_analysis.json")
    parser.add_argument("--rgb-json", default="outputs/metrics/rgb_risk_transformer_train2000_test_proposal_results.json")
    parser.add_argument("--fused-json", default="outputs/metrics/rgb_depth_risk_transformer_train2000_test_proposal_results.json")
    parser.add_argument("--comparison-output", default="outputs/plots/rgb_vs_rgbdepth_early_anticipation_comparison.png")
    parser.add_argument("--gain-output", default="outputs/plots/rgb_vs_rgbdepth_early_anticipation_gain.png")
    parser.add_argument("--summary-output", default="outputs/plots/rgb_vs_rgbdepth_proposal_summary.png")
    args = parser.parse_args()

    analysis = load_json(Path(args.analysis_json))
    rgb_metrics = load_json(Path(args.rgb_json))
    fused_metrics = load_json(Path(args.fused_json))

    labels, rgb_vals, fused_vals, gains = extract_window_metrics(analysis)
    plot_window_comparison(labels, rgb_vals, fused_vals, Path(args.comparison_output))
    plot_gain(labels, gains, Path(args.gain_output))
    plot_proposal_summary(rgb_metrics, fused_metrics, Path(args.summary_output))

    print(
        {
            "comparison_plot": args.comparison_output,
            "gain_plot": args.gain_output,
            "summary_plot": args.summary_output,
        }
    )


if __name__ == "__main__":
    main()
