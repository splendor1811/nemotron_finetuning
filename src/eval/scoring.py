"""Category-aware scoring using match functions from categories.py."""

from __future__ import annotations

import logging
from typing import List

from src.data.categories import get_match_fn
from src.inference.answer_extract import normalize_answer

logger = logging.getLogger(__name__)


def score_single(prediction: str, ground_truth: str, category: str) -> bool:
    """Score a single prediction against ground truth.

    Normalizes both prediction and ground truth using category-specific
    rules, then applies the category's match function.

    Args:
        prediction: Model's predicted answer (already extracted, may not be normalized).
        ground_truth: Ground truth answer string.
        category: Category name for choosing normalization and match function.

    Returns:
        True if prediction matches ground truth.
    """
    norm_pred = normalize_answer(prediction, category)
    norm_gt = normalize_answer(ground_truth, category)

    match_fn = get_match_fn(category)
    return match_fn(norm_pred, norm_gt)


def score_batch(
    predictions: list[str],
    ground_truths: list[str],
    categories: list[str],
) -> list[bool]:
    """Score a batch of predictions.

    Args:
        predictions: List of predicted answer strings.
        ground_truths: List of ground truth answer strings.
        categories: List of category names, aligned with predictions.

    Returns:
        List of booleans indicating correct (True) or incorrect (False).
    """
    assert len(predictions) == len(ground_truths) == len(categories), (
        f"Length mismatch: {len(predictions)} predictions, "
        f"{len(ground_truths)} ground truths, {len(categories)} categories"
    )

    results = []
    for pred, gt, cat in zip(predictions, ground_truths, categories):
        results.append(score_single(pred, gt, cat))

    correct = sum(results)
    total = len(results)
    logger.info(f"Batch scoring: {correct}/{total} correct ({correct/total:.2%})")

    return results
