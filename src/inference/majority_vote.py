"""Majority voting over multiple generations with category-aware clustering."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Tuple

from src.data.categories import CATEGORIES, get_match_fn
from src.inference.answer_extract import extract_answer

logger = logging.getLogger(__name__)


def majority_vote(
    responses: list[str],
    category: str,
) -> tuple[str, float, dict]:
    """Extract answers from all responses, normalize, and vote.

    For numeric categories (float match): cluster answers by approximate equality
    using the match function from categories.py so the tolerance is consistent
    with scoring.

    For string categories: group by exact normalized string.

    Args:
        responses: List of raw model responses for a single prompt.
        category: Category name for extraction and matching.

    Returns:
        Tuple of (winning_answer, confidence, vote_distribution) where
        confidence is the fraction of votes for the winner and
        vote_distribution maps normalized answers to vote counts.
    """
    if not responses:
        return "", 0.0, {}

    # Extract and normalize all answers
    answers = [extract_answer(r, category) for r in responses]
    # Filter out empty answers
    answers = [a for a in answers if a]

    if not answers:
        return "", 0.0, {}

    match_fn = get_match_fn(category)
    cat_spec = CATEGORIES.get(category)
    is_numeric = cat_spec is not None and cat_spec.answer_type == "float"

    if is_numeric:
        # Cluster by approximate equality
        clusters: list[tuple[str, int]] = []  # (representative, count)
        for ans in answers:
            matched = False
            for i, (rep, count) in enumerate(clusters):
                if match_fn(ans, rep):
                    clusters[i] = (rep, count + 1)
                    matched = True
                    break
            if not matched:
                clusters.append((ans, 1))

        # Find the cluster with the most votes
        clusters.sort(key=lambda x: x[1], reverse=True)
        winner, winner_count = clusters[0]
        total = sum(c for _, c in clusters)
        distribution = {rep: cnt for rep, cnt in clusters}
        confidence = winner_count / total if total > 0 else 0.0
        return winner, confidence, distribution
    else:
        # Exact match after normalization
        counts = Counter(answers)
        winner, winner_count = counts.most_common(1)[0]
        total = sum(counts.values())
        confidence = winner_count / total if total > 0 else 0.0
        return winner, confidence, dict(counts)


def run_majority_voting(
    all_responses: list[list[str]],
    categories: list[str],
) -> list[str]:
    """Run majority voting for a batch of prompts.

    Args:
        all_responses: List of lists, where all_responses[i] contains
            the multiple generations for prompt i.
        categories: Category name for each prompt, aligned with all_responses.

    Returns:
        List of winning answers, one per prompt.
    """
    assert len(all_responses) == len(categories), (
        f"Mismatch: {len(all_responses)} response sets vs {len(categories)} categories"
    )

    results = []
    for i, (responses, category) in enumerate(zip(all_responses, categories)):
        winner, confidence, distribution = majority_vote(responses, category)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Prompt {i}: category={category}, winner='{winner}', "
                f"confidence={confidence:.2%}, n_unique={len(distribution)}"
            )
        results.append(winner)

    logger.info(
        f"Majority voting complete for {len(results)} prompts, "
        f"avg responses per prompt: "
        f"{sum(len(r) for r in all_responses) / len(all_responses):.1f}"
    )
    return results
