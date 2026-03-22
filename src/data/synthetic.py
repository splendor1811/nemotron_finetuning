"""Synthetic data generation for the 6 rule-inference task categories.

Each generator produces problems in the exact same prompt format as the
competition training data, with known ground-truth answers and reasoning traces.
"""

from __future__ import annotations

import logging
import random
import string
from typing import Dict, List

import pandas as pd

from src.data.categories import CATEGORY_NAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary for text encryption tasks
# ---------------------------------------------------------------------------
WORD_VOCAB = [
    "alice", "queen", "king", "knight", "wizard", "princess", "dragon",
    "castle", "garden", "forest", "village", "mountain", "valley", "river",
    "bridge", "tower", "palace", "mirror", "crystal", "shadow", "secret",
    "golden", "silver", "magical", "clever", "brave", "hidden", "ancient",
    "door", "book", "sword", "crown", "throne", "potion", "spell",
    "cat", "dog", "mouse", "bird", "rabbit", "hatter", "student",
    "the", "under", "inside", "near", "follows", "finds", "reads",
    "creates", "discovers", "imagines", "watches", "explores", "chases",
    "sees", "dreams", "seeks", "guards", "opens", "draws",
]

# ---------------------------------------------------------------------------
# Roman numeral helpers
# ---------------------------------------------------------------------------
_ROMAN_PAIRS = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def _int_to_roman(num: int) -> str:
    """Convert a positive integer to a Roman numeral string."""
    parts: list[str] = []
    for value, symbol in _ROMAN_PAIRS:
        while num >= value:
            parts.append(symbol)
            num -= value
    return "".join(parts)


# ---------------------------------------------------------------------------
# Bit manipulation helpers
# ---------------------------------------------------------------------------

def _xor_mask(bits: str, mask: str) -> str:
    return "".join(str(int(a) ^ int(b)) for a, b in zip(bits, mask))


def _rotate_left(bits: str, k: int) -> str:
    k = k % 8
    return bits[k:] + bits[:k]


def _rotate_right(bits: str, k: int) -> str:
    k = k % 8
    return bits[8 - k:] + bits[:8 - k]


def _bit_reverse(bits: str) -> str:
    return bits[::-1]


def _bit_not(bits: str) -> str:
    return "".join("1" if b == "0" else "0" for b in bits)


def _swap_nibbles(bits: str) -> str:
    return bits[4:] + bits[:4]


def _random_8bit(rng: random.Random) -> str:
    return "".join(rng.choice("01") for _ in range(8))


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_bit_manipulation(n_samples: int, seed: int) -> List[Dict]:
    """Generate bit manipulation problems.

    Rules implemented:
    - XOR with random 8-bit mask
    - Circular left rotation by k bits
    - Circular right rotation by k bits
    - Bit reversal
    - NOT (complement)
    - Swap nibbles (swap upper and lower 4 bits)

    Each problem has 8-10 input->output examples + 1 test query.
    """
    rng = random.Random(seed)
    results: list[dict] = []

    rule_funcs = [
        ("xor_mask", lambda bits, params: _xor_mask(bits, params["mask"])),
        ("rotate_left", lambda bits, params: _rotate_left(bits, params["k"])),
        ("rotate_right", lambda bits, params: _rotate_right(bits, params["k"])),
        ("bit_reverse", lambda bits, params: _bit_reverse(bits)),
        ("bit_not", lambda bits, params: _bit_not(bits)),
        ("swap_nibbles", lambda bits, params: _swap_nibbles(bits)),
    ]

    for i in range(n_samples):
        rule_name, rule_fn = rng.choice(rule_funcs)

        # Generate rule parameters
        params: dict = {}
        if rule_name == "xor_mask":
            params["mask"] = _random_8bit(rng)
        elif rule_name in ("rotate_left", "rotate_right"):
            params["k"] = rng.randint(1, 7)

        n_examples = rng.randint(8, 10)
        examples = []
        seen_inputs: set[str] = set()
        for _ in range(n_examples + 1):  # +1 for test query
            while True:
                inp = _random_8bit(rng)
                if inp not in seen_inputs:
                    seen_inputs.add(inp)
                    break
            out = rule_fn(inp, params)
            examples.append((inp, out))

        demo_examples = examples[:-1]
        test_input, test_output = examples[-1]

        # Build prompt in exact competition format
        lines = [
            "In Alice's Wonderland, a secret bit manipulation rule transforms "
            "8-bit binary numbers. The transformation involves operations like "
            "bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority "
            "or choice functions.",
            "",
            "Here are some examples of input -> output:",
        ]
        for inp, out in demo_examples:
            lines.append(f"{inp} -> {out}")
        lines.append("")
        lines.append(f"Now, determine the output for: {test_input}")

        prompt = "\n".join(lines)

        # Build reasoning trace
        reasoning = (
            f"Let me analyze the input-output pairs to find the bit manipulation rule.\n"
            f"I'll examine each example carefully, looking for patterns in how bits "
            f"are transformed — shifts, rotations, XOR, AND, OR, NOT operations.\n"
        )
        if rule_name == "xor_mask":
            reasoning += f"Comparing inputs and outputs, I notice each output is the input XORed with a fixed mask.\n"
            reasoning += f"The mask is: {params['mask']}\n"
        elif rule_name == "rotate_left":
            reasoning += f"The output is a circular left rotation of the input by {params['k']} positions.\n"
        elif rule_name == "rotate_right":
            reasoning += f"The output is a circular right rotation of the input by {params['k']} positions.\n"
        elif rule_name == "bit_reverse":
            reasoning += "The output is the input with bits reversed (MSB becomes LSB and vice versa).\n"
        elif rule_name == "bit_not":
            reasoning += "The output is the bitwise NOT (complement) of the input.\n"
        elif rule_name == "swap_nibbles":
            reasoning += "The output swaps the upper 4 bits with the lower 4 bits of the input.\n"
        reasoning += f"Applying this rule to the test input gives: {test_output}"

        results.append({
            "id": f"synth_bit_{i:05d}",
            "prompt": prompt,
            "answer": test_output,
            "category": "bit_manipulation",
            "reasoning_trace": reasoning,
        })

    return results


def generate_gravitational_constant(n_samples: int, seed: int) -> List[Dict]:
    """Generate gravity problems.

    Pick a random g between 1.0 and 20.0.
    Generate time values, compute d = 0.5 * g * t^2.
    Round to 2 decimal places.
    """
    rng = random.Random(seed)
    results: list[dict] = []

    for i in range(n_samples):
        g = round(rng.uniform(1.0, 20.0), 2)
        n_examples = rng.randint(3, 5)

        # Generate unique time values for examples + test
        times = []
        seen: set[float] = set()
        for _ in range(n_examples + 1):
            while True:
                t = round(rng.uniform(0.5, 5.0), 2)
                if t not in seen:
                    seen.add(t)
                    break
            times.append(t)

        demo_times = times[:-1]
        test_t = times[-1]

        # Build prompt
        lines = [
            "In Alice's Wonderland, the gravitational constant has been "
            "secretly changed. Here are some example observations:",
        ]
        for t in demo_times:
            d = round(0.5 * g * t * t, 2)
            lines.append(f"For t = {t}s, distance = {d} m")
        lines.append(
            f"Now, determine the falling distance for t = {test_t}s "
            f"given d = 0.5*g*t^2."
        )

        prompt = "\n".join(lines)
        answer = str(round(0.5 * g * test_t * test_t, 2))

        reasoning = (
            f"I need to find the secret gravitational constant from the given observations.\n"
            f"Using the formula d = 0.5 * g * t^2, I can solve for g from each example.\n"
        )
        for t in demo_times:
            d = round(0.5 * g * t * t, 2)
            g_est = round(2 * d / (t * t), 2)
            reasoning += f"From t={t}, d={d}: g = 2*{d}/{t}^2 = {g_est}\n"
        reasoning += f"The gravitational constant is g = {g}\n"
        reasoning += f"For t = {test_t}: d = 0.5 * {g} * {test_t}^2 = {answer}"

        results.append({
            "id": f"synth_grav_{i:05d}",
            "prompt": prompt,
            "answer": answer,
            "category": "gravitational_constant",
            "reasoning_trace": reasoning,
        })

    return results


def generate_unit_conversion(n_samples: int, seed: int) -> List[Dict]:
    """Generate unit conversion problems.

    Pick a random conversion factor between 0.1 and 10.0.
    Generate input values, compute output = input * factor.
    """
    rng = random.Random(seed)
    results: list[dict] = []

    for i in range(n_samples):
        factor = round(rng.uniform(0.1, 10.0), 4)
        n_examples = rng.randint(3, 5)

        values = []
        seen: set[float] = set()
        for _ in range(n_examples + 1):
            while True:
                v = round(rng.uniform(1.0, 50.0), 2)
                if v not in seen:
                    seen.add(v)
                    break
            values.append(v)

        demo_values = values[:-1]
        test_value = values[-1]

        # Build prompt
        lines = [
            "In Alice's Wonderland, a secret unit conversion is applied "
            "to measurements. For example:",
        ]
        for v in demo_values:
            converted = round(v * factor, 2)
            lines.append(f"{v} m becomes {converted}")
        lines.append(f"Now, convert the following measurement: {test_value} m")

        prompt = "\n".join(lines)
        answer = str(round(test_value * factor, 2))

        reasoning = (
            f"I need to discover the hidden unit conversion factor.\n"
            f"By dividing each output by its corresponding input, I can find the conversion ratio.\n"
        )
        for v in demo_values:
            converted = round(v * factor, 2)
            ratio = round(converted / v, 4)
            reasoning += f"{converted} / {v} = {ratio}\n"
        reasoning += f"The conversion factor is approximately {factor}\n"
        reasoning += f"Applying the conversion factor to the test input gives: {answer}"

        results.append({
            "id": f"synth_unit_{i:05d}",
            "prompt": prompt,
            "answer": answer,
            "category": "unit_conversion",
            "reasoning_trace": reasoning,
        })

    return results


def generate_text_encryption(n_samples: int, seed: int) -> List[Dict]:
    """Generate substitution cipher problems.

    Create random letter permutation (a-z -> shuffled a-z).
    Encrypt words using the permutation.
    Provide encrypted->decrypted examples.
    """
    rng = random.Random(seed)
    results: list[dict] = []

    for i in range(n_samples):
        # Create a random substitution cipher (decryption mapping)
        letters = list(string.ascii_lowercase)
        shuffled = list(letters)
        rng.shuffle(shuffled)
        # encrypt_map: plaintext -> ciphertext
        encrypt_map = {p: c for p, c in zip(letters, shuffled)}
        # decrypt_map: ciphertext -> plaintext
        decrypt_map = {c: p for p, c in encrypt_map.items()}

        def encrypt_word(word: str) -> str:
            return "".join(encrypt_map.get(ch, ch) for ch in word)

        def decrypt_word(word: str) -> str:
            return "".join(decrypt_map.get(ch, ch) for ch in word)

        n_examples = rng.randint(3, 5)
        # Generate phrases (2-5 words each)
        all_phrases = []
        for _ in range(n_examples + 1):
            n_words = rng.randint(2, 5)
            phrase_words = rng.sample(WORD_VOCAB, n_words)
            all_phrases.append(phrase_words)

        demo_phrases = all_phrases[:-1]
        test_phrase = all_phrases[-1]

        # Build prompt
        lines = [
            "In Alice's Wonderland, secret encryption rules are used on "
            "text. Here are some examples:",
        ]
        for words in demo_phrases:
            encrypted = " ".join(encrypt_word(w) for w in words)
            plaintext = " ".join(words)
            lines.append(f"{encrypted} -> {plaintext}")
        encrypted_test = " ".join(encrypt_word(w) for w in test_phrase)
        lines.append(f"Now, decrypt the following text: {encrypted_test}")

        prompt = "\n".join(lines)
        answer = " ".join(test_phrase)

        reasoning = (
            f"I need to find the encryption/decryption rule from the examples.\n"
            f"Let me compare each encrypted word with its decrypted counterpart "
            f"to find the letter-by-letter transformation.\n"
            f"This appears to be a substitution cipher. Let me map each letter.\n"
            f"Decrypting the test text gives: {answer}"
        )

        results.append({
            "id": f"synth_text_{i:05d}",
            "prompt": prompt,
            "answer": answer,
            "category": "text_encryption",
            "reasoning_trace": reasoning,
        })

    return results


def generate_numeral_system(n_samples: int, seed: int) -> List[Dict]:
    """Generate decimal to Roman numeral conversion problems."""
    rng = random.Random(seed)
    results: list[dict] = []

    for i in range(n_samples):
        n_examples = rng.randint(3, 5)

        # Pick numbers for examples + test (range 1 to 3999)
        numbers = []
        seen: set[int] = set()
        for _ in range(n_examples + 1):
            while True:
                n = rng.randint(1, 3999)
                if n not in seen:
                    seen.add(n)
                    break
            numbers.append(n)

        demo_numbers = numbers[:-1]
        test_number = numbers[-1]

        # Build prompt
        lines = [
            "In Alice's Wonderland, numbers are secretly converted into a "
            "different numeral system. Some examples are given below:",
        ]
        for n in demo_numbers:
            lines.append(f"{n} -> {_int_to_roman(n)}")
        lines.append(
            f"Now, write the number {test_number} in the Wonderland numeral system."
        )

        prompt = "\n".join(lines)
        answer = _int_to_roman(test_number)

        reasoning = (
            f"I need to understand the numeral system conversion rule.\n"
            f"Looking at the examples, I can see the pattern of how numbers are converted.\n"
            f"This appears to be converting to Roman numerals.\n"
        )
        for n in demo_numbers:
            reasoning += f"{n} -> {_int_to_roman(n)}\n"
        reasoning += f"Applying the conversion to {test_number} gives: {answer}"

        results.append({
            "id": f"synth_num_{i:05d}",
            "prompt": prompt,
            "answer": answer,
            "category": "numeral_system",
            "reasoning_trace": reasoning,
        })

    return results


def generate_equation_transformation(n_samples: int, seed: int) -> List[Dict]:
    """Generate equation transformation problems.

    Create a random symbol mapping and apply to short expressions.
    Symbols include digits, operators, and special characters.
    """
    rng = random.Random(seed)
    results: list[dict] = []

    # Pool of symbols to use in equations
    symbol_pool = list("0123456789+-*/=()abcxyz<>|\\{}[]'\"!@#$%^&`~?")

    for i in range(n_samples):
        # Create a random character mapping (subset of symbols)
        n_mapped = rng.randint(6, 15)
        from_symbols = rng.sample(symbol_pool, n_mapped)
        to_symbols = rng.sample(symbol_pool, n_mapped)
        char_map = dict(zip(from_symbols, to_symbols))

        def transform(expr: str) -> str:
            return "".join(char_map.get(ch, ch) for ch in expr)

        n_examples = rng.randint(3, 4)
        # Generate random expressions
        all_exprs = []
        for _ in range(n_examples + 1):
            expr_len = rng.randint(3, 6)
            # Build expression from mapped symbols so transformation is meaningful
            expr = "".join(rng.choice(from_symbols) for _ in range(expr_len))
            all_exprs.append(expr)

        demo_exprs = all_exprs[:-1]
        test_expr = all_exprs[-1]

        # Build prompt
        lines = [
            "In Alice's Wonderland, a secret set of transformation rules is "
            "applied to equations. Below are a few examples:",
        ]
        for expr in demo_exprs:
            lines.append(f"{expr} = {transform(expr)}")
        lines.append(f"Now, determine the result for: {test_expr}")

        prompt = "\n".join(lines)
        answer = transform(test_expr)

        reasoning = (
            f"I need to find the transformation rules applied to equations.\n"
            f"Let me compare input and output equations to identify the substitution pattern.\n"
            f"I can see how symbols are mapped:\n"
        )
        for f, t in char_map.items():
            reasoning += f"  '{f}' -> '{t}'\n"
        reasoning += f"Applying these rules to the test equation gives: {answer}"

        results.append({
            "id": f"synth_eq_{i:05d}",
            "prompt": prompt,
            "answer": answer,
            "category": "equation_transformation",
            "reasoning_trace": reasoning,
        })

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_GENERATORS = {
    "bit_manipulation": generate_bit_manipulation,
    "gravitational_constant": generate_gravitational_constant,
    "unit_conversion": generate_unit_conversion,
    "text_encryption": generate_text_encryption,
    "numeral_system": generate_numeral_system,
    "equation_transformation": generate_equation_transformation,
}


def generate_all_synthetic(n_per_category: int, seed: int) -> pd.DataFrame:
    """Generate balanced synthetic data across all 6 categories.

    Args:
        n_per_category: Number of samples per category.
        seed: Base random seed (each category gets seed + offset for
              reproducibility).

    Returns:
        DataFrame with columns [id, prompt, answer, category, reasoning_trace].
    """
    all_records: list[dict] = []

    for offset, (cat_name, gen_fn) in enumerate(_GENERATORS.items()):
        cat_seed = seed + offset * 1000
        logger.info(
            f"Generating {n_per_category} samples for '{cat_name}' "
            f"(seed={cat_seed})"
        )
        records = gen_fn(n_per_category, cat_seed)
        all_records.extend(records)
        logger.info(f"  -> {len(records)} samples generated")

    df = pd.DataFrame(all_records)
    logger.info(
        f"Total synthetic samples: {len(df)}\n"
        f"Category distribution:\n{df['category'].value_counts().to_string()}"
    )
    return df
