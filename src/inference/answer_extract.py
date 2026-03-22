"""Extract and normalize answers from model responses with <think> reasoning traces."""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def extract_thinking_and_answer(response: str) -> tuple[str, str]:
    """Split response into (thinking_trace, final_answer).

    Handles cases where:
    - Model uses <think>...</think> block
    - Model uses multiple <think> blocks (take the last answer portion)
    - Model does not use <think> tags at all (entire response is answer)
    - <think> block is empty
    """
    if not response:
        return "", ""

    # Match the <think>...</think> block (greedy to capture all thinking)
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    matches = pattern.findall(response)

    if matches:
        # Remove all <think>...</think> blocks to get the answer
        answer = pattern.sub("", response).strip()
        # Combine all thinking blocks
        thinking = "\n".join(m.strip() for m in matches)
        return thinking, answer
    else:
        # No thinking tags -- entire response is the answer
        return "", response.strip()


def normalize_answer(answer: str, category: str) -> str:
    """Category-specific normalization of extracted answers.

    Args:
        answer: Raw extracted answer string.
        category: One of the category names from categories.py.

    Returns:
        Normalized answer string ready for matching.
    """
    if not answer:
        return ""

    if category == "bit_manipulation":
        # Strip whitespace, find the first 8-char binary substring
        answer = answer.strip()
        match = re.search(r"[01]{8}", answer)
        if match:
            return match.group(0)
        # Fallback: strip and return whatever we have
        return answer.strip()

    elif category in ("gravitational_constant", "unit_conversion"):
        # Strip whitespace, extract first numeric value (possibly negative, with decimals/exponents)
        answer = answer.strip()
        match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", answer)
        if match:
            return match.group(0)
        return answer.strip()

    elif category == "text_encryption":
        # Strip, lowercase for case-insensitive comparison
        return answer.strip().lower()

    elif category == "numeral_system":
        # Strip, uppercase (Roman numerals are uppercase by convention)
        return answer.strip().upper()

    elif category == "equation_transformation":
        # Strip whitespace only -- expressions need exact formatting
        return answer.strip()

    else:
        # Unknown category -- just strip
        logger.warning(f"Unknown category '{category}' for normalization, stripping only")
        return answer.strip()


def extract_answer(response: str, category: str) -> str:
    """Full pipeline: extract answer from response, then normalize.

    Args:
        response: Full model response (may include <think> block).
        category: Category name for normalization.

    Returns:
        Normalized answer string.
    """
    _, raw_answer = extract_thinking_and_answer(response)
    return normalize_answer(raw_answer, category)
