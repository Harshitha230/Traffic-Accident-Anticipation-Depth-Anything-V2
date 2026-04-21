import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import get_config
from model_risk_transformer import DepthRiskTransformer
from risk_metrics import regression_metrics
from risk_sequence_dataset import RiskSequenceDataset
from utils import load_checkpoint, save_json


def evaluate_loader(
    model,
    loader,
    device: str,
    tta_threshold: float = 0.5,
    fps: float | None = None,
) -> tuple[dict, list[dict]]:
    model.eval()
    all_targets = []
    all_predictions = []
    all_video_labels = []
    all_onset_indices = []
    prediction_rows: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            targets = batch["risk_curve"].to(device)
            predictions = model(features)

            target_curves = targets.detach().cpu().numpy()
            predicted_curves = predictions.detach().cpu().numpy()
            video_labels = batch["video_label"].detach().cpu().numpy()
            onset_indices = batch["accident_onset_index"].detach().cpu().numpy()

            all_targets.append(target_curves)
            all_predictions.append(predicted_curves)
            all_video_labels.append(video_labels)
            all_onset_indices.append(onset_indices)

            for sample_id, video_id, label, onset_index, target_curve, predicted_curve in zip(
                batch["sample_id"],
                batch["video_id"],
                video_labels,
                onset_indices,
                target_curves,
                predicted_curves,
            ):
                prediction_rows.append(
                    {
                        "sample_id": sample_id,
                        "video_id": video_id,
                        "video_label": int(label),
                        "accident_onset_index": int(onset_index),
                        "target_risk_curve": target_curve.tolist(),
                        "predicted_risk_curve": predicted_curve.tolist(),
                        "max_target_risk": float(target_curve.max()),
                        "max_predicted_risk": float(predicted_curve.max()),
                        "max_predicted_risk_pre_onset": float(
                            predicted_curve[: max(1, min(int(onset_index), len(predicted_curve)) if int(onset_index) >= 0 else len(predicted_curve))].max()
                        ),
                    }
                )

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_predictions, axis=0)
    video_labels = np.concatenate(all_video_labels, axis=0)
    onset_indices = np.concatenate(all_onset_indices, axis=0)
    metrics = regression_metrics(
        y_true,
        y_pred,
        video_labels=video_labels,
        onset_indices=onset_indices,
        tta_threshold=tta_threshold,
        fps=fps,
    )
    return metrics, prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--feature-mode", choices=["depth", "rgb", "rgb_depth"], default=None)
    parser.add_argument("--include-depth-delta", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tta-threshold", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--output-prefix", type=str, default="depth_risk_transformer_eval")
    args = parser.parse_args()

    config = get_config()
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    csv_path = Path(args.csv_path) if args.csv_path else config.test_split_path

    checkpoint = load_checkpoint(Path(args.checkpoint_path), map_location=config.device)
    checkpoint_config = checkpoint.get("config", {})
    feature_mode = args.feature_mode or checkpoint.get("feature_mode") or checkpoint_config.get("feature_mode", "depth")
    include_depth_delta = args.include_depth_delta or checkpoint.get("include_depth_delta", False)

    dataset = RiskSequenceDataset(
        csv_path,
        config,
        feature_mode=feature_mode,
        include_depth_delta=include_depth_delta,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model_hparams = checkpoint.get("model_hparams", {})
    input_dim = int(checkpoint["input_dim"])
    model = DepthRiskTransformer(
        input_dim=input_dim,
        hidden_dim=model_hparams.get("hidden_dim", 128),
        num_heads=model_hparams.get("num_heads", 4),
        num_layers=model_hparams.get("num_layers", 3),
        dropout=model_hparams.get("dropout", 0.1),
        sequence_length=config.sequence_length,
    ).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics, prediction_rows = evaluate_loader(
        model,
        loader,
        config.device,
        tta_threshold=args.tta_threshold,
        fps=args.fps,
    )
    metrics_path = config.outputs_dir / "metrics" / f"{args.output_prefix}_results.json"
    predictions_path = config.outputs_dir / "predictions" / f"{args.output_prefix}_predictions.csv"
    save_json(metrics, metrics_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    print(metrics)


if __name__ == "__main__":
    main()
