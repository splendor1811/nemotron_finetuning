"""Data loading, category detection, and train/val splitting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DataConfig, ExperimentConfig, PathsConfig
from src.data.categories import CATEGORY_NAMES, detect_category

logger = logging.getLogger(__name__)


def load_raw_data(
    paths: PathsConfig,
    data_config: DataConfig,
    split: str = "train",
) -> pd.DataFrame:
    """
    Load raw CSV and add auto-detected category column.

    Args:
        paths: Resolved paths config.
        data_config: Data config with file names.
        split: "train" or "test".

    Returns:
        DataFrame with columns [id, prompt, answer (if train), category].
    """
    filename = data_config.train_file if split == "train" else data_config.test_file
    filepath = Path(paths.raw_data_dir) / filename

    logger.info(f"Loading {split} data from {filepath}")
    df = pd.read_csv(filepath)

    # Auto-detect categories
    df["category"] = df["prompt"].apply(detect_category)

    unknown = df[df["category"] == "unknown"]
    if len(unknown) > 0:
        logger.warning(f"{len(unknown)} rows with unknown category")

    # Log category distribution
    counts = df["category"].value_counts()
    logger.info(f"Category distribution:\n{counts.to_string()}")

    return df


def create_train_val_split(
    df: pd.DataFrame,
    val_fraction: float = 0.1,
    strategy: str = "stratified",
    seed: int = 3407,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and validation sets.

    Args:
        df: Full training DataFrame with 'category' column.
        val_fraction: Fraction of data for validation.
        strategy: "stratified" (by category) or "random".
        seed: Random seed.

    Returns:
        (train_df, val_df) tuple.
    """
    if val_fraction <= 0:
        return df, pd.DataFrame(columns=df.columns)

    stratify = df["category"] if strategy == "stratified" else None

    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        random_state=seed,
        stratify=stratify,
    )

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")
    logger.info(f"Val category distribution:\n{val_df['category'].value_counts().to_string()}")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_and_split(config: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: load raw data and split in one call."""
    df = load_raw_data(config.paths, config.data, split="train")
    return create_train_val_split(
        df,
        val_fraction=config.data.val_fraction,
        strategy=config.data.val_strategy,
        seed=config.project.seed,
    )
