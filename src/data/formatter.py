"""Format training data into chat template for SFT and GRPO."""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset

from src.data.categories import CATEGORIES


def _build_reasoning_trace(prompt: str, answer: str, category: str) -> str:
    """
    Generate a synthetic reasoning trace for a training example.

    Since train.csv only contains final answers (no chain-of-thought),
    we create template-based reasoning that demonstrates the rule inference process.
    """
    if category == "bit_manipulation":
        return (
            f"Let me analyze the input-output pairs to find the bit manipulation rule.\n"
            f"I'll examine each example carefully, looking for patterns in how bits "
            f"are transformed — shifts, rotations, XOR, AND, OR, NOT operations.\n"
            f"After studying all the examples, I can identify the transformation pattern.\n"
            f"Applying this rule to the test input gives: {answer}"
        )
    elif category == "gravitational_constant":
        return (
            f"I need to find the secret gravitational constant from the given observations.\n"
            f"Using the formula d = 0.5 * g * t^2, I can solve for g from each example.\n"
            f"Let me compute g from the provided time-distance pairs and find the consistent value.\n"
            f"Using this gravitational constant to compute the answer: {answer}"
        )
    elif category == "unit_conversion":
        return (
            f"I need to discover the hidden unit conversion factor.\n"
            f"By dividing each output by its corresponding input, I can find the conversion ratio.\n"
            f"Let me check this ratio is consistent across all examples.\n"
            f"Applying the conversion factor to the test input gives: {answer}"
        )
    elif category == "text_encryption":
        return (
            f"I need to find the encryption/decryption rule from the examples.\n"
            f"Let me compare each encrypted word with its decrypted counterpart "
            f"to find the letter-by-letter transformation.\n"
            f"This appears to be a substitution cipher. Let me map each letter.\n"
            f"Decrypting the test text gives: {answer}"
        )
    elif category == "numeral_system":
        return (
            f"I need to understand the numeral system conversion rule.\n"
            f"Looking at the examples, I can see the pattern of how numbers are converted.\n"
            f"This appears to be converting to Roman numerals or a similar system.\n"
            f"Applying the conversion to the test number gives: {answer}"
        )
    elif category == "equation_transformation":
        return (
            f"I need to find the transformation rules applied to equations.\n"
            f"Let me compare input and output equations to identify the substitution pattern.\n"
            f"I can see how symbols are mapped or operations are transformed.\n"
            f"Applying these rules to the test equation gives: {answer}"
        )
    else:
        return f"Analyzing the examples to find the hidden rule.\nThe answer is: {answer}"


def format_row_for_sft(
    row: Dict[str, Any],
    include_reasoning: bool = True,
) -> List[Dict[str, str]]:
    """
    Convert a data row into chat messages for SFT training.

    Args:
        row: Dict with 'prompt', 'answer', 'category' keys.
        include_reasoning: Whether to include <think> reasoning trace.

    Returns:
        List of message dicts with 'role' and 'content'.
    """
    category = row.get("category", "unknown")
    prompt = row["prompt"]
    answer = str(row["answer"])

    if include_reasoning:
        system_msg = "detailed thinking on"
        trace = _build_reasoning_trace(prompt, answer, category)
        assistant_content = f"<think>\n{trace}\n</think>\n\n{answer}"
    else:
        system_msg = "detailed thinking off"
        assistant_content = answer

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_content},
    ]
    return messages


def format_row_for_grpo(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a data row into GRPO format (prompt only, no assistant response).

    Returns dict with 'prompt' (list of messages), 'answer', 'category'.
    """
    return {
        "prompt": [
            {"role": "system", "content": "detailed thinking on"},
            {"role": "user", "content": row["prompt"]},
        ],
        "answer": str(row["answer"]),
        "category": row.get("category", "unknown"),
    }


def build_sft_dataset(
    df: pd.DataFrame,
    tokenizer: Any,
    reasoning_ratio: float = 0.8,
    seed: int = 3407,
) -> Dataset:
    """
    Convert DataFrame to HuggingFace Dataset formatted for SFT training.

    Args:
        df: DataFrame with columns [id, prompt, answer, category].
        tokenizer: Tokenizer with apply_chat_template method.
        reasoning_ratio: Fraction of examples with reasoning traces.
        seed: Random seed for reasoning/direct split.

    Returns:
        HF Dataset with 'text' column (formatted chat strings).
    """
    rng = random.Random(seed)
    texts = []

    for _, row in df.iterrows():
        include_reasoning = rng.random() < reasoning_ratio
        messages = format_row_for_sft(row.to_dict(), include_reasoning=include_reasoning)

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    return Dataset.from_dict({"text": texts})


def build_grpo_dataset(df: pd.DataFrame) -> Dataset:
    """
    Convert DataFrame to HuggingFace Dataset formatted for GRPO training.

    Returns HF Dataset with columns: prompt (list of dicts), answer, category.
    """
    records = [format_row_for_grpo(row.to_dict()) for _, row in df.iterrows()]
    return Dataset.from_list(records)
