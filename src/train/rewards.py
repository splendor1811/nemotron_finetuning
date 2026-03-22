"""Reward functions for GRPO training following TRL GRPOTrainer interface.

Each reward function receives:
- completions: list of list of dicts [{"role": "assistant", "content": "..."}]
- Additional dataset columns as keyword arguments (answer, category, etc.)
- Returns: list of floats (one reward per completion)
"""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)


def _get_completion_text(completion: list[dict]) -> str:
    """Extract the assistant content string from a single completion."""
    for msg in completion:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    # Fallback: return content of last message
    if completion:
        return completion[-1].get("content", "")
    return ""


def correctness_reward(
    completions: list[list[dict]],
    answer: list[str],
    category: list[str],
    **kwargs,
) -> list[float]:
    """Primary reward: extract answer from completion and compare to ground truth.

    Uses extract_answer() and score_single() from existing modules.
    Returns 1.0 if correct, 0.0 if wrong.

    Args:
        completions: List of completions, each a list of message dicts.
        answer: List of ground truth answer strings (from dataset column).
        category: List of category strings (from dataset column).

    Returns:
        List of float rewards (1.0 or 0.0).
    """
    from src.inference.answer_extract import extract_answer
    from src.eval.scoring import score_single

    rewards = []
    for completion, gt, cat in zip(completions, answer, category):
        try:
            text = _get_completion_text(completion)
            predicted = extract_answer(text, cat)
            is_correct = score_single(predicted, gt, cat)
            rewards.append(1.0 if is_correct else 0.0)
        except Exception as e:
            logger.warning(f"Error in correctness_reward: {e}")
            rewards.append(0.0)

    return rewards


def format_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward proper <think>...</think> structure.

    Incremental scoring:
    - +0.25 for <think> tag present
    - +0.25 for </think> tag present
    - +0.25 for non-empty content between tags
    - +0.25 for answer content after </think>
    - -0.5 penalty for malformed output (multiple <think> blocks, nested tags)

    Args:
        completions: List of completions, each a list of message dicts.

    Returns:
        List of float rewards in [0.0, 1.0].
    """
    rewards = []
    for completion in completions:
        try:
            text = _get_completion_text(completion)
            reward = 0.0

            has_open = "<think>" in text
            has_close = "</think>" in text

            if has_open:
                reward += 0.25
            if has_close:
                reward += 0.25

            # Check for content between tags
            if has_open and has_close:
                match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
                if match:
                    thinking_content = match.group(1).strip()
                    if len(thinking_content) > 0:
                        reward += 0.25

                # Check for answer after </think>
                close_idx = text.rfind("</think>")
                after_close = text[close_idx + len("</think>"):].strip()
                if len(after_close) > 0:
                    reward += 0.25

            # Penalty for malformed structure
            open_count = text.count("<think>")
            close_count = text.count("</think>")
            if open_count > 1 or close_count > 1:
                reward = max(0.0, reward - 0.5)

            # Penalty if close comes before open
            if has_open and has_close:
                if text.index("</think>") < text.index("<think>"):
                    reward = max(0.0, reward - 0.5)

            rewards.append(reward)
        except Exception as e:
            logger.warning(f"Error in format_reward: {e}")
            rewards.append(0.0)

    return rewards


def reasoning_quality_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward structured reasoning inside <think> blocks.

    Scoring criteria:
    - +0.2 for containing step-by-step indicator words
    - +0.2 for identifying pattern/rule (pattern-related keywords)
    - +0.2 for applying to test input (application-related keywords)
    - -0.2 for extremely short thinking (<50 chars)
    - -0.1 for extremely long thinking (>3000 chars)

    Args:
        completions: List of completions, each a list of message dicts.

    Returns:
        List of float rewards.
    """
    step_words = re.compile(
        r"\b(step|first|second|third|then|next|therefore|finally|"
        r"so|because|since|thus|hence|let me|let\'s|I will|I need to)\b",
        re.IGNORECASE,
    )
    pattern_words = re.compile(
        r"\b(pattern|rule|relationship|observe|notice|see that|"
        r"consistent|each|every|always|formula|conversion|factor|"
        r"cipher|shift|XOR|transformation|mapping)\b",
        re.IGNORECASE,
    )
    apply_words = re.compile(
        r"\b(apply|applying|test input|result|answer is|gives us|"
        r"therefore the answer|computing|calculate|convert|"
        r"decrypt|transform|output)\b",
        re.IGNORECASE,
    )

    rewards = []
    for completion in completions:
        try:
            text = _get_completion_text(completion)
            reward = 0.0

            # Extract thinking content
            match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            if match:
                thinking = match.group(1).strip()
            else:
                # No think block: base reward stays 0
                rewards.append(0.0)
                continue

            # Step-by-step reasoning
            if step_words.search(thinking):
                reward += 0.2

            # Pattern/rule identification
            if pattern_words.search(thinking):
                reward += 0.2

            # Application to test input
            if apply_words.search(thinking):
                reward += 0.2

            # Length penalties
            if len(thinking) < 50:
                reward -= 0.2
            if len(thinking) > 3000:
                reward -= 0.1

            rewards.append(max(0.0, reward))
        except Exception as e:
            logger.warning(f"Error in reasoning_quality_reward: {e}")
            rewards.append(0.0)

    return rewards
