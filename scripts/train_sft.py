#!/usr/bin/env python3
"""CLI entry point for SFT training pipeline."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SFT fine-tuning on Nemotron with LoRA via Unsloth"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/lora_r16.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config overrides as key.subkey=value (can be repeated)",
    )
    args = parser.parse_args()

    logger.info(f"Config: {args.config}")
    if args.override:
        logger.info(f"Overrides: {args.override}")

    from src.train.sft import run_sft

    run_sft(config_path=args.config, overrides=args.override or None)


if __name__ == "__main__":
    main()
