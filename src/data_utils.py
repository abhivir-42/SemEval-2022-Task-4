"""
Data utilities for SemEval 2022 Task 4 - PCL Detection.
Handles loading, preprocessing, and dataset creation.
"""

import pandas as pd
import numpy as np  # noqa: F401
from pathlib import Path
from sklearn.model_selection import train_test_split


# === Paths (relative to project root) ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TSV_PATH = DATA_DIR / "dontpatronizeme_pcl.tsv"
TRAIN_SPLIT = DATA_DIR / "practice-splits" / "train_semeval_parids-labels.csv"
DEV_SPLIT = DATA_DIR / "practice-splits" / "dev_semeval_parids-labels.csv"
TEST_PATH = DATA_DIR / "test" / "task4_test.tsv"


def load_pcl_dataset(tsv_path=TSV_PATH):
    """Load the full Don't Patronize Me dataset, skipping the 4-line disclaimer."""
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        skiprows=4,
        header=None,
        names=["par_id", "art_id", "keyword", "country", "text", "label"],
    )
    df["par_id"] = df["par_id"].astype(str)
    # Binary label: 0-1 → 0 (No PCL), 2-4 → 1 (PCL)
    df["binary_label"] = (df["label"] >= 2).astype(int)
    return df


def load_split_ids(split_path):
    """Load paragraph IDs and (optionally) task2 labels from a split file."""
    split_df = pd.read_csv(split_path)
    split_df["par_id"] = split_df["par_id"].astype(str)
    return split_df


def load_test_set(test_path=TEST_PATH):
    """Load the official test set (no labels)."""
    df = pd.read_csv(
        test_path,
        sep="\t",
        header=None,
        names=["par_id", "art_id", "keyword", "country", "text"],
    )
    df["par_id"] = df["par_id"].astype(str)
    return df


def format_community_aware(text, keyword):
    """Prepend community keyword context to the text."""
    return f"[Community: {keyword}] {text}"


def get_train_dev_data(community_aware=True):
    """
    Load and prepare train and dev datasets.

    Returns:
        train_df: DataFrame with columns [par_id, keyword, text, label]
        dev_df: DataFrame with columns [par_id, keyword, text, label]
    """
    full_df = load_pcl_dataset()
    train_ids = load_split_ids(TRAIN_SPLIT)
    dev_ids = load_split_ids(DEV_SPLIT)

    # Join to get text and labels
    train_df = full_df[full_df["par_id"].isin(train_ids["par_id"])].copy()
    dev_df = full_df[full_df["par_id"].isin(dev_ids["par_id"])].copy()

    # Preserve dev order from split file
    dev_df = dev_ids[["par_id"]].merge(dev_df, on="par_id", how="left")

    if community_aware:
        train_df["input_text"] = train_df.apply(
            lambda r: format_community_aware(r["text"], r["keyword"]), axis=1
        )
        dev_df["input_text"] = dev_df.apply(
            lambda r: format_community_aware(r["text"], r["keyword"]), axis=1
        )
    else:
        train_df["input_text"] = train_df["text"]
        dev_df["input_text"] = dev_df["text"]

    train_out = train_df[["par_id", "keyword", "input_text", "binary_label"]].rename(
        columns={"binary_label": "label"}
    )
    dev_out = dev_df[["par_id", "keyword", "input_text", "binary_label"]].rename(
        columns={"binary_label": "label"}
    )

    return train_out, dev_out


def get_test_data(community_aware=True):
    """Load and prepare test dataset (no labels)."""
    test_df = load_test_set()

    if community_aware:
        test_df["input_text"] = test_df.apply(
            lambda r: format_community_aware(r["text"], r["keyword"]), axis=1
        )
    else:
        test_df["input_text"] = test_df["text"]

    return test_df[["par_id", "keyword", "input_text"]]


def create_train_val_split(train_df, val_ratio=0.15, seed=42):
    """
    Split training data into internal train/validation sets.
    Stratified by label to preserve class distribution.
    """
    train_split, val_split = train_test_split(
        train_df,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_df["label"],
    )
    return train_split.reset_index(drop=True), val_split.reset_index(drop=True)


if __name__ == "__main__":
    # Quick sanity check
    train_df, dev_df = get_train_dev_data()
    test_df = get_test_data()

    print(f"Train: {len(train_df)} samples")
    print(f"  Positive: {train_df['label'].sum()} ({train_df['label'].mean():.1%})")
    print(f"  Negative: {(train_df['label'] == 0).sum()}")
    print(f"Dev: {len(dev_df)} samples")
    print(f"  Positive: {dev_df['label'].sum()} ({dev_df['label'].mean():.1%})")
    print(f"Test: {len(test_df)} samples")
    print(f"\nSample input text:")
    print(train_df["input_text"].iloc[0][:200])
