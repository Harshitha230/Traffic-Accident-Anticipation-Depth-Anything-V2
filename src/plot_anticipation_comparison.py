import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir


def load_analysis(path: Path) -> dict:
    return json.loads(path.read_text())


def extract_rows(analysis: dict, subset: str = "all") -> tuple[list[str], list[float], list[float], list[float]]:
    keys = [
        ("pre_onset_min_gap_0", "Pre-onset"),
        ("pre_onset_min_gap_5", ">=5 frames"),
        ("pre_onset_min_gap_10", ">=10 frames"),
    ]

    labels = []
    rgb_values = []
    fusion_values = []
    gains = []

    for key, label in keys:
        entry = analysis[key][subset]
        labels.append(label)
        rgb_values.append(float(entry["rgb"]["auc_roc_pre_onset"]))
        fusion_values.append(float(entry["rgb_depth"]["auc_roc_pre_onset"]))
        gains.append(float(entry["auc_gain_rgb_depth_minus_rgb"]))

    return labels, rgb_values, fusion_values, gains


def plot_comparison(labels: list[str], rgb_values: list[float], fusion_values: list[float], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8, 5))
    rgb_bars = ax.bar(x - width / 2, rgb_values, width, label="RGB-only", color="#4C78A8")
    fusion_bars = ax.bar(x + width / 2, fusion_values, width, label="RGB+Depth", color="#E45756")

    ax.set_title("Early Anticipation Performance")
    ax.set_ylabel("AUC-ROC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.55, 1.0)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bars in (rgb_bars, fusion_bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_gain(labels: list[str], gains: list[float], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, gains, width=0.5, color="#54A24B")

    ax.set_title("Depth Contribution to Early Anticipation")
    ax.set_ylabel("AUC Gain over RGB-only")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, max(gains) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"+{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-json",
        type=str,
        default="outputs/metrics/rgb_vs_rgbdepth_anticipation_subset_analysis.json",
    )
    parser.add_argument("--subset", type=str, default="all")
    parser.add_argument(
        "--comparison-output",
        type=str,
        default="outputs/plots/rgb_vs_rgbdepth_early_anticipation_comparison.png",
    )
    parser.add_argument(
        "--gain-output",
        type=str,
        default="outputs/plots/rgb_vs_rgbdepth_early_anticipation_gain.png",
    )
    args = parser.parse_args()

    analysis = load_analysis(Path(args.analysis_json))
    labels, rgb_values, fusion_values, gains = extract_rows(analysis, subset=args.subset)

    plot_comparison(labels, rgb_values, fusion_values, Path(args.comparison_output))
    plot_gain(labels, gains, Path(args.gain_output))

    print(
        {
            "comparison_plot": str(Path(args.comparison_output)),
            "gain_plot": str(Path(args.gain_output)),
            "labels": labels,
            "rgb": rgb_values,
            "rgb_depth": fusion_values,
            "gains": gains,
        }
    )


if __name__ == "__main__":
    main()
