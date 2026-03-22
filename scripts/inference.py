#!/usr/bin/env python3
"""Generate a Kaggle submission CSV using majority voting.

Usage:
    python scripts/inference.py --model-path outputs/checkpoints/merged \
        --input data/raw/test.csv \
        --output outputs/submissions/submission.csv \
        --num-generations 64
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, resolve_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission")
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml", help="Config file path"
    )
    parser.add_argument(
        "--model-path", required=True, help="Path to model checkpoint or HF model ID"
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional path to LoRA adapter (if not using merged model)",
    )
    parser.add_argument(
        "--input",
        default="data/raw/test.csv",
        help="Path to test CSV (default: data/raw/test.csv)",
    )
    parser.add_argument(
        "--output",
        default="outputs/submissions/submission.csv",
        help="Path to output submission CSV",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=64,
        help="Number of generations for majority voting (default: 64)",
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

    # Override num_generations from CLI if specified
    if args.num_generations != config.inference.num_generations:
        config.inference.num_generations = args.num_generations
        logger.info(f"Overriding num_generations to {args.num_generations}")

    # Resolve input/output paths
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load test data
    import pandas as pd

    logger.info(f"Loading test data from {input_path}")
    test_df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(test_df)} test samples")

    # Detect categories
    from src.data.categories import detect_category

    if "category" not in test_df.columns:
        test_df["category"] = test_df["prompt"].apply(detect_category)

    prompts = test_df["prompt"].tolist()
    categories = test_df["category"].tolist()

    logger.info(f"Category distribution:\n{test_df['category'].value_counts().to_string()}")

    # Run inference
    from src.inference.engine import NemotronEngine

    engine = NemotronEngine(
        model_path=args.model_path,
        config=config.inference,
        adapter_path=args.adapter_path,
    )

    num_generations = config.inference.num_generations
    logger.info(f"Running inference with {num_generations} generations per prompt")

    all_generations = engine.generate_batch(
        prompts, num_generations=num_generations
    )

    # Run majority voting
    from src.inference.majority_vote import run_majority_voting

    predictions = run_majority_voting(all_generations, categories)

    # Build submission
    submission_df = pd.DataFrame({
        "id": test_df["id"],
        "answer": predictions,
    })

    # Validate no empty answers
    empty_count = submission_df["answer"].apply(lambda x: x == "" or pd.isna(x)).sum()
    if empty_count > 0:
        logger.warning(
            f"{empty_count} empty predictions in submission. "
            "These may be scored as incorrect."
        )

    submission_df.to_csv(output_path, index=False)
    logger.info(f"Saved submission ({len(submission_df)} rows) to {output_path}")

    # Print summary
    logger.info("\n=== Submission Summary ===")
    logger.info(f"Total predictions: {len(submission_df)}")
    logger.info(f"Empty predictions: {empty_count}")
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
