import pandas as pd
from sklearn.model_selection import train_test_split

from config import get_config
from utils import ensure_dir


def main() -> None:
    config = get_config()
    ensure_dir(config.splits_dir)

    metadata = pd.read_csv(config.metadata_path)
    official_train_df = metadata[metadata["split"] == "train"].copy()
    official_test_df = metadata[metadata["split"] == "test"].copy()

    train_df, val_df = train_test_split(
        official_train_df,
        test_size=config.val_ratio,
        random_state=config.seed,
        stratify=official_train_df["video_label"]
        if len(official_train_df["video_label"].unique()) > 1
        else None,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = official_test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    train_df.to_csv(config.train_split_path, index=False)
    val_df.to_csv(config.val_split_path, index=False)
    test_df.to_csv(config.test_split_path, index=False)

    print(f"Train split: {len(train_df)} samples")
    print(f"Val split: {len(val_df)} samples")
    print(f"Test split: {len(test_df)} samples")
    print("\nLabel counts:")
    print("train")
    print(train_df["video_label"].value_counts().sort_index())
    print("val")
    print(val_df["video_label"].value_counts().sort_index())
    print("test")
    print(test_df["video_label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
