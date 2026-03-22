#!/usr/bin/env python3
"""Prepare training data: load CSV, detect categories, split, format, save."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, resolve_paths
from src.data.loader import load_raw_data, create_train_val_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config file path")
    parser.add_argument("--override", action="append", default=[], help="Config overrides (key=value)")
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.override)
    config = resolve_paths(config)

    # Load and split
    df = load_raw_data(config.paths, config.data, split="train")
    train_df, val_df = create_train_val_split(
        df,
        val_fraction=config.data.val_fraction,
        strategy=config.data.val_strategy,
        seed=config.project.seed,
    )

    # Save processed DataFrames
    output_dir = Path(config.paths.processed_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    logger.info(f"Saved train ({len(train_df)} rows) to {train_path}")
    logger.info(f"Saved val ({len(val_df)} rows) to {val_path}")

    # Also save the test data with categories
    test_df = load_raw_data(config.paths, config.data, split="test")
    test_path = output_dir / "test.csv"
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved test ({len(test_df)} rows) to {test_path}")

    # Summary
    logger.info("\n=== Data Preparation Complete ===")
    logger.info(f"Train categories:\n{train_df['category'].value_counts().to_string()}")
    logger.info(f"Val categories:\n{val_df['category'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
