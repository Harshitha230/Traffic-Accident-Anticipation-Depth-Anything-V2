import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from config import get_config


def parse_label_sequence(label_text: str) -> list[int]:
    return list(ast.literal_eval(label_text))


def build_risk_curve(frame_labels: list[int], sigma: float) -> list[float]:
    if not frame_labels:
        return []

    if sum(frame_labels) == 0:
        return [0.0] * len(frame_labels)

    label_array = np.asarray(frame_labels, dtype=float)
    smoothed = gaussian_filter1d(label_array, sigma=sigma, mode="nearest")
    max_value = float(smoothed.max())
    if max_value > 0:
        smoothed = smoothed / max_value
    return [round(float(value), 6) for value in smoothed.tolist()]


def enrich_dataframe(df: pd.DataFrame, sigma: float) -> pd.DataFrame:
    df = df.copy()

    accident_frame_counts = []
    accident_onsets = []
    risk_curves = []
    max_risks = []

    for label_text in df["frame_labels"]:
        frame_labels = parse_label_sequence(label_text)
        risk_curve = build_risk_curve(frame_labels, sigma=sigma)
        accident_frame_counts.append(sum(frame_labels))
        accident_onsets.append(next((idx for idx, value in enumerate(frame_labels) if value == 1), -1))
        risk_curves.append(json.dumps(risk_curve))
        max_risks.append(max(risk_curve) if risk_curve else 0.0)

    df["accident_frame_count"] = accident_frame_counts
    df["accident_onset_index"] = accident_onsets
    df["risk_curve"] = risk_curves
    df["max_risk"] = max_risks
    return df


def process_file(csv_path: Path, sigma: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    enriched = enrich_dataframe(df, sigma=sigma)
    enriched.to_csv(csv_path, index=False)
    return enriched


def main() -> None:
    config = get_config()

    targets = [
        config.metadata_path,
        config.train_split_path,
        config.val_split_path,
        config.test_split_path,
    ]

    for csv_path in targets:
        enriched = process_file(csv_path, sigma=config.gaussian_sigma)
        print(
            f"{csv_path.name}: rows={len(enriched)}, "
            f"positive_rows={(enriched['video_label'] == 1).sum()}, "
            f"avg_max_risk={enriched['max_risk'].mean():.4f}"
        )


if __name__ == "__main__":
    main()
