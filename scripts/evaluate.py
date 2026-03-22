#!/usr/bin/env python3
"""Evaluate a model checkpoint on the validation set.

Usage:
    python scripts/evaluate.py --model-path outputs/checkpoints/merged \
        --config configs/base.yaml \
        --override inference.num_generations=64
"""

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Evaluate model on validation set")
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
        "--override",
        action="append",
        default=[],
        help="Config overrides (key=value)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save evaluation results (default: outputs/eval)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config, overrides=args.override)
    config = resolve_paths(config)

    output_dir = Path(args.output_dir) if args.output_dir else Path(config.paths.output_dir) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Init wandb
    if not args.no_wandb:
        try:
            import wandb

            wandb.init(
                project=config.project.wandb_project,
                entity=config.project.wandb_entity,
                name=f"eval-{Path(args.model_path).name}",
                config=config.model_dump(),
                tags=["eval"],
            )
        except Exception as e:
            logger.warning(f"Failed to init wandb: {e}")

    # Load validation data
    import pandas as pd

    val_path = Path(config.paths.processed_data_dir) / "val.csv"
    if not val_path.exists():
        logger.error(
            f"Validation file not found at {val_path}. "
            "Run 'python scripts/prepare_data.py' first."
        )
        sys.exit(1)

    val_df = pd.read_csv(val_path)
    logger.info(f"Loaded {len(val_df)} validation samples from {val_path}")

    prompts = val_df["prompt"].tolist()
    ground_truths = val_df["answer"].tolist()
    categories = val_df["category"].tolist()

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

    # Score and compute metrics
    from src.eval.metrics import compute_metrics, log_metrics_to_wandb

    metrics = compute_metrics(
        predictions=predictions,
        ground_truths=[str(gt) for gt in ground_truths],
        categories=categories,
        all_generations=all_generations,
    )

    # Save results
    results_path = output_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {results_path}")

    # Save detailed predictions
    val_df["prediction"] = predictions
    val_df["correct"] = [
        metrics["overall"]["correct"] > 0  # placeholder
        for _ in range(len(val_df))
    ]
    # Re-score individually for the correct column
    from src.eval.scoring import score_batch

    val_df["correct"] = score_batch(
        predictions, [str(gt) for gt in ground_truths], categories
    )
    predictions_path = output_dir / "predictions.csv"
    val_df.to_csv(predictions_path, index=False)
    logger.info(f"Saved predictions to {predictions_path}")

    # Log to wandb
    if not args.no_wandb:
        log_metrics_to_wandb(metrics, step=0, prefix="eval")

    logger.info("Evaluation complete!")
    logger.info(f"Overall accuracy: {metrics['overall']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
