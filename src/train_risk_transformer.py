import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from config import get_config
from model_risk_transformer import DepthRiskTransformer
from risk_metrics import regression_metrics
from risk_sequence_dataset import get_sequence_dataloaders
from utils import plot_training_history, save_checkpoint, save_json, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_targets = []
    all_predictions = []
    all_video_labels = []

    for batch in tqdm(loader, leave=False):
        features = batch["features"].to(device)
        targets = batch["risk_curve"].to(device)
        video_labels = batch["video_label"].detach().cpu().numpy()

        if training:
            optimizer.zero_grad()

        predictions = model(features)
        loss = criterion(predictions, targets)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * features.size(0)
        all_targets.append(targets.detach().cpu().numpy())
        all_predictions.append(predictions.detach().cpu().numpy())
        all_video_labels.append(video_labels)

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_predictions, axis=0)
    video_labels = np.concatenate(all_video_labels, axis=0)
    metrics = regression_metrics(y_true, y_pred, video_labels=video_labels)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=str, default=None)
    parser.add_argument("--val-csv", type=str, default=None)
    parser.add_argument("--feature-mode", choices=["depth", "rgb", "rgb_depth"], default="depth")
    parser.add_argument("--include-depth-delta", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-name", type=str, default="depth_risk_transformer_full")
    args = parser.parse_args()

    config = get_config()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.num_workers = args.num_workers
    set_seed(config.seed)

    train_csv_path = Path(args.train_csv) if args.train_csv else config.train_split_path
    val_csv_path = Path(args.val_csv) if args.val_csv else config.val_split_path

    train_loader, val_loader, _ = get_sequence_dataloaders(
        config,
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        test_csv_path=None,
        feature_mode=args.feature_mode,
        include_depth_delta=args.include_depth_delta,
    )

    sample_batch = next(iter(train_loader))
    input_dim = int(sample_batch["features"].shape[-1])

    model = DepthRiskTransformer(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        sequence_length=config.sequence_length,
    ).to(config.device)

    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
        "train_auc_roc": [],
        "val_auc_roc": [],
    }
    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        train_metrics = run_epoch(model, train_loader, criterion, config.device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, config.device, optimizer=None)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_mae"].append(train_metrics["mae"])
        history["val_mae"].append(val_metrics["mae"])
        history["train_auc_roc"].append(train_metrics.get("auc_roc", 0.0))
        history["val_auc_roc"].append(val_metrics.get("auc_roc", 0.0))

        print(
            f"Epoch {epoch + 1}/{config.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_mae={val_metrics['mae']:.4f} "
            f"val_auc={val_metrics.get('auc_roc', 0.0):.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint_path = config.checkpoints_dir / f"{args.run_name}.pt"
            save_checkpoint(
                {
                    "model_name": "depth_risk_transformer",
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "config": config.__dict__,
                    "model_hparams": {
                        "hidden_dim": args.hidden_dim,
                        "num_heads": args.num_heads,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                    },
                    "feature_mode": args.feature_mode,
                    "include_depth_delta": args.include_depth_delta,
                },
                checkpoint_path,
            )

    plot_training_history(history, config.outputs_dir / "plots" / f"{args.run_name}_history.png")
    save_json(history, config.outputs_dir / "metrics" / f"{args.run_name}_history.json")


if __name__ == "__main__":
    main()
