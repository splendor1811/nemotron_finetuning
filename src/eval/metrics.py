"""Compute evaluation metrics: accuracy, per-category breakdown, pass@1, maj@K."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.data.categories import CATEGORY_NAMES
from src.eval.scoring import score_single
from src.inference.majority_vote import majority_vote

logger = logging.getLogger(__name__)


def compute_metrics(
    predictions: list[str],
    ground_truths: list[str],
    categories: list[str],
    all_generations: Optional[list[list[str]]] = None,
) -> dict:
    """Compute overall and per-category evaluation metrics.

    Args:
        predictions: Final predicted answers (e.g. after majority voting).
        ground_truths: Ground truth answer strings.
        categories: Category for each sample.
        all_generations: If provided, list of all raw generation lists per sample
            for computing pass@1 and maj@K metrics.

    Returns:
        Nested dict with structure:
        {
            "overall": {"accuracy": float, "total": int, "correct": int},
            "per_category": {
                "category_name": {"accuracy": float, "total": int, "correct": int},
                ...
            },
            "pass_at_1": float (if all_generations provided),
            "maj_at_K": {"K": int, "accuracy": float} (if all_generations provided),
        }
    """
    assert len(predictions) == len(ground_truths) == len(categories)

    # --- Overall and per-category accuracy ---
    correct_flags = [
        score_single(pred, gt, cat)
        for pred, gt, cat in zip(predictions, ground_truths, categories)
    ]

    total = len(correct_flags)
    correct = sum(correct_flags)

    cat_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for flag, cat in zip(correct_flags, categories):
        cat_stats[cat]["total"] += 1
        if flag:
            cat_stats[cat]["correct"] += 1

    per_category = {}
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        per_category[cat] = {
            "accuracy": s["correct"] / s["total"] if s["total"] > 0 else 0.0,
            "correct": s["correct"],
            "total": s["total"],
        }

    metrics: Dict[str, Any] = {
        "overall": {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
        },
        "per_category": per_category,
    }

    # --- pass@1 and maj@K from raw generations ---
    if all_generations is not None:
        assert len(all_generations) == total

        # pass@1: for each sample, score the *first* generation only
        pass1_correct = 0
        for gens, gt, cat in zip(all_generations, ground_truths, categories):
            if gens:
                from src.inference.answer_extract import extract_answer

                first_answer = extract_answer(gens[0], cat)
                if score_single(first_answer, gt, cat):
                    pass1_correct += 1

        metrics["pass_at_1"] = pass1_correct / total if total > 0 else 0.0

        # maj@K: majority vote over all generations (already done for predictions,
        # but compute confidence stats)
        k_values = set()
        total_confidence = 0.0
        for gens, gt, cat in zip(all_generations, ground_truths, categories):
            k_values.add(len(gens))
            _, confidence, _ = majority_vote(gens, cat)
            total_confidence += confidence

        k = max(k_values) if k_values else 0
        metrics["maj_at_K"] = {
            "K": k,
            "accuracy": metrics["overall"]["accuracy"],  # already majority-voted
            "avg_confidence": total_confidence / total if total > 0 else 0.0,
        }

    # Log summary
    logger.info(f"Overall accuracy: {metrics['overall']['accuracy']:.4f} "
                f"({metrics['overall']['correct']}/{metrics['overall']['total']})")
    for cat, s in per_category.items():
        logger.info(f"  {cat}: {s['accuracy']:.4f} ({s['correct']}/{s['total']})")
    if "pass_at_1" in metrics:
        logger.info(f"pass@1: {metrics['pass_at_1']:.4f}")
    if "maj_at_K" in metrics:
        m = metrics["maj_at_K"]
        logger.info(f"maj@{m['K']}: {m['accuracy']:.4f} (avg confidence: {m['avg_confidence']:.4f})")

    return metrics


def log_metrics_to_wandb(metrics: dict, step: int, prefix: str = "eval") -> None:
    """Log metrics dict to wandb with proper nesting.

    Args:
        metrics: Output of compute_metrics().
        step: Global step for wandb logging.
        prefix: Prefix for metric keys (e.g. "eval", "test").
    """
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed, skipping metric logging")
        return

    if wandb.run is None:
        logger.warning("No active wandb run, skipping metric logging")
        return

    flat: Dict[str, Any] = {}

    # Overall
    if "overall" in metrics:
        for k, v in metrics["overall"].items():
            flat[f"{prefix}/overall/{k}"] = v

    # Per-category
    if "per_category" in metrics:
        for cat, cat_metrics in metrics["per_category"].items():
            for k, v in cat_metrics.items():
                flat[f"{prefix}/category/{cat}/{k}"] = v

    # pass@1
    if "pass_at_1" in metrics:
        flat[f"{prefix}/pass_at_1"] = metrics["pass_at_1"]

    # maj@K
    if "maj_at_K" in metrics:
        for k, v in metrics["maj_at_K"].items():
            flat[f"{prefix}/maj_at_K/{k}"] = v

    wandb.log(flat, step=step)
    logger.info(f"Logged {len(flat)} metrics to wandb at step {step}")
