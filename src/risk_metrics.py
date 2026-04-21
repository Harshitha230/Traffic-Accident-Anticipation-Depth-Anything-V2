import numpy as np
from sklearn.metrics import roc_auc_score


def _safe_prefix_end(onset_index: int, sequence_length: int) -> int:
    if onset_index < 0:
        return sequence_length
    return max(1, min(int(onset_index), sequence_length))


def _video_scores(y_pred: np.ndarray, onset_indices=None, pre_onset_only: bool = False) -> np.ndarray:
    if not pre_onset_only or onset_indices is None:
        return y_pred.max(axis=1)

    onset_indices = np.asarray(onset_indices, dtype=int)
    scores = []
    for curve, onset_index in zip(y_pred, onset_indices):
        prefix_end = _safe_prefix_end(int(onset_index), len(curve))
        scores.append(float(np.max(curve[:prefix_end])))
    return np.asarray(scores, dtype=np.float32)


def _tta_metrics(y_pred: np.ndarray, video_labels, onset_indices, threshold: float, fps: float | None = None) -> dict:
    video_labels = np.asarray(video_labels, dtype=int)
    onset_indices = np.asarray(onset_indices, dtype=int)

    warning_distances = []
    warned_count = 0
    positive_count = 0

    for curve, label, onset_index in zip(y_pred, video_labels, onset_indices):
        if int(label) != 1 or int(onset_index) <= 0:
            continue

        positive_count += 1
        prefix_end = _safe_prefix_end(int(onset_index), len(curve))
        pre_onset_curve = curve[:prefix_end]
        crossing_indices = np.flatnonzero(pre_onset_curve >= threshold)
        if crossing_indices.size == 0:
            continue

        warned_count += 1
        first_warning_index = int(crossing_indices[0])
        warning_distances.append(int(onset_index) - first_warning_index)

    metrics = {
        f"tta_threshold_{threshold:.2f}_positive_count": positive_count,
        f"tta_threshold_{threshold:.2f}_warned_count": warned_count,
        f"tta_threshold_{threshold:.2f}_warning_recall": float(warned_count / positive_count) if positive_count else 0.0,
        f"tta_threshold_{threshold:.2f}_frames": float(np.mean(warning_distances)) if warning_distances else 0.0,
    }
    if fps is not None and fps > 0:
        metrics[f"tta_threshold_{threshold:.2f}_seconds"] = metrics[f"tta_threshold_{threshold:.2f}_frames"] / fps
    return metrics


def regression_metrics(
    y_true_curves,
    y_pred_curves,
    video_labels=None,
    onset_indices=None,
    tta_threshold: float | None = None,
    fps: float | None = None,
) -> dict:
    y_true = np.asarray(y_true_curves, dtype=np.float32)
    y_pred = np.asarray(y_pred_curves, dtype=np.float32)

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    metrics = {
        "mse": mse,
        "mae": mae,
    }

    if video_labels is not None:
        video_labels = np.asarray(video_labels)
        video_scores = _video_scores(y_pred)
        if len(np.unique(video_labels)) > 1:
            metrics["auc_roc"] = float(roc_auc_score(video_labels, video_scores))
        else:
            metrics["auc_roc"] = 0.0

        if onset_indices is not None:
            pre_onset_scores = _video_scores(y_pred, onset_indices=onset_indices, pre_onset_only=True)
            if len(np.unique(video_labels)) > 1:
                metrics["auc_roc_pre_onset"] = float(roc_auc_score(video_labels, pre_onset_scores))
            else:
                metrics["auc_roc_pre_onset"] = 0.0

            if tta_threshold is not None:
                metrics.update(_tta_metrics(y_pred, video_labels, onset_indices, tta_threshold, fps=fps))

    return metrics
