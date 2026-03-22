"""RLVR post-training pipeline using GRPO with verifiable rewards."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from src.config import ExperimentConfig, load_config, resolve_paths
from src.data.formatter import build_grpo_dataset

logger = logging.getLogger(__name__)


def create_grpo_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    config: ExperimentConfig,
) -> Any:
    """Create a GRPOTrainer with reward functions and hyperparameters from config.

    Args:
        model: Model with LoRA adapters applied (from SFT checkpoint).
        tokenizer: Tokenizer.
        train_dataset: HF Dataset with columns: prompt, answer, category.
        config: Full experiment config (uses config.rlvr for GRPO settings).

    Returns:
        Configured GRPOTrainer instance (not yet trained).
    """
    from trl import GRPOTrainer, GRPOConfig

    from src.train.rewards import (
        correctness_reward,
        format_reward,
        reasoning_quality_reward,
    )

    rc = config.rlvr
    output_dir = str(Path(config.paths.checkpoint_dir) / "rlvr")

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=rc.learning_rate,
        per_device_train_batch_size=rc.per_device_train_batch_size,
        gradient_accumulation_steps=rc.gradient_accumulation_steps,
        num_generations=rc.num_generations,
        max_prompt_length=rc.max_prompt_length,
        max_completion_length=rc.max_completion_length,
        temperature=rc.temperature,
        num_train_epochs=rc.num_train_epochs,
        beta=rc.kl_coef,
        max_grad_norm=rc.max_grad_norm,
        warmup_ratio=rc.warmup_ratio,
        lr_scheduler_type=rc.lr_scheduler_type,
        bf16=rc.bf16,
        report_to=rc.report_to,
        save_steps=rc.save_steps,
        logging_steps=rc.logging_steps,
        save_total_limit=3,
        seed=config.project.seed,
        run_name=f"rlvr-{config.project.name}",
    )

    # Reward functions and weights
    reward_funcs = [
        correctness_reward,
        format_reward,
        reasoning_quality_reward,
    ]
    reward_weights = [
        rc.reward_weights.correctness,
        rc.reward_weights.format,
        rc.reward_weights.reasoning_quality,
    ]

    logger.info(f"GRPO reward functions: {[f.__name__ for f in reward_funcs]}")
    logger.info(f"GRPO reward weights: {reward_weights}")
    logger.info(
        f"GRPO config: lr={rc.learning_rate}, num_generations={rc.num_generations}, "
        f"kl_coef={rc.kl_coef}, temperature={rc.temperature}"
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    return trainer


def run_rlvr(
    config_path: str,
    sft_checkpoint: str,
    overrides: Optional[List[str]] = None,
) -> None:
    """Full RLVR post-training pipeline.

    Steps:
        1. Load and resolve config
        2. Set random seeds
        3. Initialize wandb with "rlvr-" prefix
        4. Load SFT model from checkpoint
        5. Build GRPO dataset from processed data
        6. Create GRPOTrainer
        7. Train
        8. Save model

    Args:
        config_path: Path to YAML experiment config.
        sft_checkpoint: Path to SFT checkpoint directory to load from.
        overrides: Optional list of "key.subkey=value" override strings.
    """
    import torch

    # 1. Load config
    config = load_config(config_path, overrides=overrides)
    config = resolve_paths(config)
    logger.info(f"Loaded config from {config_path}")

    # Verify SFT checkpoint exists
    sft_path = Path(sft_checkpoint)
    if not sft_path.exists():
        raise FileNotFoundError(
            f"SFT checkpoint not found at {sft_path}. "
            f"Run SFT training first."
        )
    logger.info(f"Using SFT checkpoint: {sft_path}")

    # 2. Set seeds
    seed = config.project.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed: {seed}")

    # 3. Init wandb
    try:
        import wandb

        wandb.init(
            project=config.project.wandb_project,
            entity=config.project.wandb_entity,
            name=f"rlvr-{config.project.name}",
            config=config.model_dump(),
            tags=["rlvr", "grpo"],
        )
        logger.info(f"wandb initialized: {config.project.wandb_project}")
    except ImportError:
        logger.warning("wandb not installed; skipping experiment tracking")

    # 4. Load SFT model from checkpoint
    from src.train.model import load_model_and_tokenizer, apply_lora

    # Override model name to load from SFT checkpoint
    config.model.name = str(sft_path)
    model, tokenizer = load_model_and_tokenizer(config.model)
    model = apply_lora(model, config.lora)
    logger.info("Model loaded from SFT checkpoint with LoRA applied")

    # 5. Build GRPO dataset from processed data
    processed_dir = Path(config.paths.processed_data_dir)
    train_csv = processed_dir / "train.csv"

    if not train_csv.exists():
        raise FileNotFoundError(
            f"Processed training data not found at {train_csv}. "
            f"Run scripts/prepare_data.py first."
        )

    train_df = pd.read_csv(train_csv)
    logger.info(f"Loaded {len(train_df)} training examples from {train_csv}")

    train_dataset = build_grpo_dataset(train_df)
    logger.info(f"Built GRPO dataset: {len(train_dataset)} examples")
    logger.info(f"Dataset columns: {train_dataset.column_names}")

    # 6. Create GRPOTrainer
    trainer = create_grpo_trainer(model, tokenizer, train_dataset, config)

    # 7. Train
    logger.info("Starting RLVR/GRPO training...")
    train_result = trainer.train()
    logger.info(f"Training complete. Metrics: {train_result.metrics}")

    # 8. Save model
    final_dir = Path(config.paths.checkpoint_dir) / "rlvr" / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Saved RLVR adapter to {final_dir}")

    # Log final metrics to wandb
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(train_result.metrics)
            wandb.finish()
            logger.info("wandb run finished")
    except ImportError:
        pass

    logger.info("RLVR pipeline complete.")
