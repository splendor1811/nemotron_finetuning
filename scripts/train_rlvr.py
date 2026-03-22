#!/usr/bin/env python3
"""CLI entry point for RLVR/GRPO post-training pipeline."""

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
        description="Run RLVR/GRPO post-training on SFT checkpoint with verifiable rewards"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rlvr/grpo_base.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--sft-checkpoint",
        type=str,
        required=True,
        help="Path to SFT checkpoint directory",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config overrides as key.subkey=value (can be repeated)",
    )
    args = parser.parse_args()

    logger.info(f"Config: {args.config}")
    logger.info(f"SFT checkpoint: {args.sft_checkpoint}")
    if args.override:
        logger.info(f"Overrides: {args.override}")

    from src.train.rlvr import run_rlvr

    run_rlvr(
        config_path=args.config,
        sft_checkpoint=args.sft_checkpoint,
        overrides=args.override or None,
    )


if __name__ == "__main__":
    main()
