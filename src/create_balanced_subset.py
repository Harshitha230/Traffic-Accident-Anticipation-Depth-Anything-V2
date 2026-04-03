import argparse
from pathlib import Path

import pandas as pd

from config import get_config
from utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--per-class", type=int, default=25)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = get_config()
    source_map = {
        "train": config.train_split_path,
        "val": config.val_split_path,
        "test": config.test_split_path,
    }

    df = pd.read_csv(source_map[args.split])
    seed = config.seed if args.seed is None else args.seed

    negatives = df[df["video_label"] == 0].sample(n=min(args.per_class, (df["video_label"] == 0).sum()), random_state=seed)
    positives = df[df["video_label"] == 1].sample(n=min(args.per_class, (df["video_label"] == 1).sum()), random_state=seed)

    subset = pd.concat([negatives, positives], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    ensure_dir(config.splits_dir)
    output_path = config.splits_dir / f"balanced_{args.split}_subset_{len(subset)}.csv"
    subset.to_csv(output_path, index=False)

    print(f"Saved balanced subset to {output_path}")
    print(subset["video_label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
