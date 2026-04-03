import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import get_config
from utils import ensure_dir, save_json


def build_video_features(sequence_array: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            sequence_array.mean(axis=0),
            sequence_array.std(axis=0),
            sequence_array.max(axis=0),
            sequence_array.min(axis=0),
        ]
    ).astype(np.float32)


def load_split_features(split_csv_path: Path, feature_metadata_path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    split_df = pd.read_csv(split_csv_path).copy()
    feature_df = pd.read_csv(feature_metadata_path).copy()
    merged = split_df.merge(feature_df, on=["sample_id", "split"], how="inner")
    if merged.empty:
        raise ValueError(f"No matching feature rows found for {split_csv_path}")

    feature_rows = []
    labels = []
    for _, row in merged.iterrows():
        sequence_array = np.load(row["feature_path"]).astype(np.float32)
        feature_rows.append(build_video_features(sequence_array))
        labels.append(int(row["video_label"]))

    X = np.stack(feature_rows)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, merged


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
    }


def main() -> None:
    config = get_config()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv",
        type=str,
        default=str(config.splits_dir / "balanced_train_subset_200_classifier_train.csv"),
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default=str(config.splits_dir / "balanced_train_subset_200_classifier_val.csv"),
    )
    parser.add_argument("--output-prefix", type=str, default="depth_video_logreg_200")
    args = parser.parse_args()
    train_csv_path = Path(args.train_csv)
    val_csv_path = Path(args.val_csv)

    X_train, y_train, train_meta = load_split_features(train_csv_path, config.depth_features_metadata_path)
    X_val, y_val, val_meta = load_split_features(val_csv_path, config.depth_features_metadata_path)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000)),
        ]
    )
    model.fit(X_train, y_train)

    train_probabilities = model.predict_proba(X_train)[:, 1]
    val_probabilities = model.predict_proba(X_val)[:, 1]

    train_metrics = evaluate_predictions(y_train, train_probabilities)
    val_metrics = evaluate_predictions(y_val, val_probabilities)
    metrics = {
        "train": train_metrics,
        "val": val_metrics,
    }

    model_dir = ensure_dir(config.checkpoints_dir)
    model_path = model_dir / f"{args.output_prefix}.joblib"
    joblib.dump(model, model_path)

    metrics_path = config.outputs_dir / "metrics" / f"{args.output_prefix}_results.json"
    save_json(metrics, metrics_path)

    predictions_path = config.outputs_dir / "predictions" / f"{args.output_prefix}_val_predictions.csv"
    ensure_dir(predictions_path.parent)
    prediction_df = val_meta[["sample_id", "video_id", "video_label"]].copy()
    prediction_df["predicted_probability"] = val_probabilities
    prediction_df["predicted_label"] = (val_probabilities >= 0.5).astype(int)
    prediction_df.to_csv(predictions_path, index=False)

    print({"model_path": str(model_path), "metrics": metrics, "predictions_path": str(predictions_path)})


if __name__ == "__main__":
    main()
