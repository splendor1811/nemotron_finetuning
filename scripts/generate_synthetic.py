#!/usr/bin/env python3
"""Generate synthetic training data for all 6 rule-inference categories.

Usage:
    python scripts/generate_synthetic.py --config configs/base.yaml --n-per-category 500
    python scripts/generate_synthetic.py --n-per-category 100 --seed 42 --output-dir data/synthetic
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, resolve_paths
from src.data.synthetic import generate_all_synthetic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for the Nemotron reasoning challenge"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Config file path (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--n-per-category",
        type=int,
        default=500,
        help="Number of samples per category (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed; uses config seed if not specified",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory; uses config paths.synthetic_data_dir if not specified",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config overrides (key=value)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config, overrides=args.override)
    config = resolve_paths(config)

    seed = args.seed if args.seed is not None else config.project.seed
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.paths.synthetic_data_dir)

    logger.info(f"Seed: {seed}")
    logger.info(f"Samples per category: {args.n_per_category}")
    logger.info(f"Output directory: {output_dir}")

    # Generate
    df = generate_all_synthetic(n_per_category=args.n_per_category, seed=seed)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_train.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} synthetic samples to {output_path}")

    # Also save per-category files for inspection
    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat]
        cat_path = output_dir / f"synthetic_{cat}.csv"
        cat_df.to_csv(cat_path, index=False)
        logger.info(f"  {cat}: {len(cat_df)} samples -> {cat_path}")

    logger.info("\n=== Synthetic Data Generation Complete ===")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Category distribution:\n{df['category'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
