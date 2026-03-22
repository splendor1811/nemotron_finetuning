"""Category definitions, detection, and matching functions for the 6 task types."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class CategorySpec:
    name: str
    display_name: str
    detection_patterns: List[str]
    answer_type: str  # "binary_string", "float", "text", "roman", "expression"
    system_prompt_thinking: str = "detailed thinking on"
    system_prompt_direct: str = "detailed thinking off"


def _match_exact(prediction: str, ground_truth: str) -> bool:
    return prediction.strip() == ground_truth.strip()


def _match_case_insensitive(prediction: str, ground_truth: str) -> bool:
    return prediction.strip().lower() == ground_truth.strip().lower()


def _match_float_approx(prediction: str, ground_truth: str, rel_tol: float = 1e-2) -> bool:
    try:
        pred_val = float(prediction.strip())
        gt_val = float(ground_truth.strip())
        if gt_val == 0:
            return abs(pred_val) < 1e-6
        return abs(pred_val - gt_val) / abs(gt_val) <= rel_tol
    except (ValueError, TypeError):
        return False


def get_match_fn(category: str) -> Callable[[str, str], bool]:
    """Return the appropriate matching function for a category."""
    match_fns = {
        "bit_manipulation": _match_exact,
        "gravitational_constant": _match_float_approx,
        "unit_conversion": _match_float_approx,
        "text_encryption": _match_case_insensitive,
        "numeral_system": _match_exact,
        "equation_transformation": _match_exact,
    }
    return match_fns.get(category, _match_exact)


CATEGORIES: Dict[str, CategorySpec] = {
    "bit_manipulation": CategorySpec(
        name="bit_manipulation",
        display_name="Bit Manipulation",
        detection_patterns=["bit manipulation"],
        answer_type="binary_string",
    ),
    "gravitational_constant": CategorySpec(
        name="gravitational_constant",
        display_name="Gravitational Constant",
        detection_patterns=["gravitational"],
        answer_type="float",
    ),
    "unit_conversion": CategorySpec(
        name="unit_conversion",
        display_name="Unit Conversion",
        detection_patterns=["unit conversion"],
        answer_type="float",
    ),
    "text_encryption": CategorySpec(
        name="text_encryption",
        display_name="Text Encryption",
        detection_patterns=["encryption rules", "decrypt"],
        answer_type="text",
    ),
    "numeral_system": CategorySpec(
        name="numeral_system",
        display_name="Numeral System",
        detection_patterns=["numeral system"],
        answer_type="roman",
    ),
    "equation_transformation": CategorySpec(
        name="equation_transformation",
        display_name="Equation Transformation",
        detection_patterns=["transformation rules"],
        answer_type="expression",
    ),
}

CATEGORY_NAMES = list(CATEGORIES.keys())


def detect_category(prompt: str) -> str:
    """Detect category from prompt content using keyword matching."""
    prompt_lower = prompt.lower()
    for name, spec in CATEGORIES.items():
        for pattern in spec.detection_patterns:
            if pattern in prompt_lower:
                return name
    return "unknown"
