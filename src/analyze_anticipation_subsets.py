import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils import ensure_dir, save_json


def parse_curve(text: str) -> np.ndarray:
    return np.asarray(ast.literal_eval(text), dtype=np.float32)


def pre_onset_score(curve: np.ndarray, onset_index: int, min_gap: int) -> float:
    if onset_index < 0:
        usable = curve
    else:
        usable_end = max(1, min(int(onset_index) - int(min_gap), len(curve)))
        usable = curve[:usable_end]
    return float(np.max(usable))


def subset_mask(df: pd.DataFrame, subset_name: str) -> pd.Series:
    if subset_name == "all":
        return pd.Series(True, index=df.index)
    if subset_name == "night":
        return df["lighting"].astype(str).eq("Night")
    if subset_name == "day":
        return df["lighting"].astype(str).eq("Day")
    if subset_name == "rainy":
        return df["weather"].astype(str).eq("Rainy")
    if subset_name == "snowy":
        return df["weather"].astype(str).eq("Snowy")
    if subset_name == "ego_yes":
        return df["ego_involve"].astype(str).eq("Yes")
    if subset_name == "ego_no":
        return df["ego_involve"].astype(str).eq("No")
    if subset_name == "known_conditions":
        return (
            ~df["lighting"].astype(str).eq("Unknown")
            & ~df["weather"].astype(str).eq("Unknown")
            & ~df["ego_involve"].astype(str).eq("Unknown")
        )
    raise ValueError(f"Unsupported subset: {subset_name}")


def load_predictions(predictions_path: Path, split_df: pd.DataFrame) -> pd.DataFrame:
    pred_df = pd.read_csv(predictions_path).copy()
    pred_df["sample_id"] = pred_df["sample_id"].astype(str)
    pred_df["video_id"] = pred_df["video_id"].astype(str).str.zfill(6)
    pred_df["video_label"] = pred_df["video_label"].astype(int)
    pred_df["accident_onset_index"] = pred_df["accident_onset_index"].astype(int)
    pred_df["predicted_risk_curve"] = pred_df["predicted_risk_curve"].map(parse_curve)
    merged = split_df.merge(pred_df, on=["sample_id", "video_id", "video_label", "accident_onset_index"], how="inner")
    if merged.empty:
        raise ValueError(f"No matching rows after merging predictions from {predictions_path}")
    return merged


def compute_auc(df: pd.DataFrame, min_gap: int) -> dict:
    labels = df["video_label"].astype(int).to_numpy()
    scores = np.asarray(
        [
            pre_onset_score(curve, onset_index, min_gap)
            for curve, onset_index in zip(df["predicted_risk_curve"], df["accident_onset_index"])
        ],
        dtype=np.float32,
    )
    auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.0
    return {
        "count": int(len(df)),
        "positives": int((labels == 1).sum()),
        "negatives": int((labels == 0).sum()),
        "auc_roc_pre_onset": auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", required=True, type=str)
    parser.add_argument("--baseline-predictions", required=True, type=str)
    parser.add_argument("--fusion-predictions", required=True, type=str)
    parser.add_argument("--min-gaps", nargs="+", type=int, default=[0, 5, 10])
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["all", "night", "rainy", "snowy", "ego_yes", "known_conditions"],
    )
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    split_df = pd.read_csv(args.split_csv).copy()
    split_df["sample_id"] = split_df["sample_id"].astype(str)
    split_df["video_id"] = split_df["video_id"].astype(int).astype(str).str.zfill(6)

    baseline_df = load_predictions(Path(args.baseline_predictions), split_df)
    fusion_df = load_predictions(Path(args.fusion_predictions), split_df)

    results = {}
    for min_gap in args.min_gaps:
        gap_key = f"pre_onset_min_gap_{min_gap}"
        results[gap_key] = {}
        for subset in args.subsets:
            base_mask = subset_mask(baseline_df, subset)
            fusion_mask = subset_mask(fusion_df, subset)
            base_metrics = compute_auc(baseline_df.loc[base_mask].reset_index(drop=True), min_gap=min_gap)
            fusion_metrics = compute_auc(fusion_df.loc[fusion_mask].reset_index(drop=True), min_gap=min_gap)
            results[gap_key][subset] = {
                "rgb": base_metrics,
                "rgb_depth": fusion_metrics,
                "auc_gain_rgb_depth_minus_rgb": float(
                    fusion_metrics["auc_roc_pre_onset"] - base_metrics["auc_roc_pre_onset"]
                ),
            }

    print(json.dumps(results, indent=2))

    if args.output_json is not None:
        output_path = Path(args.output_json)
        ensure_dir(output_path.parent)
        save_json(results, output_path)


if __name__ == "__main__":
    main()
