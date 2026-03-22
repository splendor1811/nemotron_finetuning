"""SFT training pipeline: dataset creation, trainer setup, and full run."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import ExperimentConfig, load_config, resolve_paths
from src.data.formatter import build_sft_dataset

logger = logging.getLogger(__name__)

_INSTRUCTION_PART = "<|im_start|>user\n"
_RESPONSE_PART_ = "<|imi_start|>assistant\n"


def _get_response_template(tokenizer: Any) -> str:
    """Extract the assistant response template from the chat template.

    For Nemotron models, the assistant turn typically starts with
    '<extra_id_1>assistant\n'. We use a minimal probe conversation to
    detect the exact string that precedes the assistant response.
    """
    probe_messages = [
        {"role": "system", "content": "detailed thinking on"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "RESPONSE_START"},
    ]
    rendered = tokenizer.apply_chat_template(
        probe_messages, tokenize=False, add_generation_prompt=False
    )

    # Find the marker and take the preceding template text
    marker = "RESPONSE_START"
    idx = rendered.find(marker)
    if idx == -1:
        raise ValueError(
            "Could not detect response template from chat template. "
            "Check that the tokenizer has a valid chat_template."
        )

    # Walk backwards from idx to find the start of the assistant turn header.
    # We look for a newline boundary that marks the header.
    prefix = rendered[:idx]
    # Take the last line(s) that form the assistant header
    # Typically this is something like "<extra_id_1>assistant\n"
    lines = prefix.split("\n")
    # The response template is the last non-empty segment before the response
    # Reconstruct: find the assistant header
    # Strategy: take text after the last user message ends
    user_end = prefix.rfind("Hello")
    if user_end != -1:
        after_user = prefix[user_end + len("Hello"):]
        response_template = after_user.strip("\n")
        # Keep trailing newline if present in original
        if after_user.endswith("\n"):
            response_template += "\n"
        logger.info(f"Detected response template: {repr(response_template)}")
        return response_template

    # Fallback: use the last 30 chars before the marker
    response_template = prefix[-30:].lstrip()
    logger.warning(f"Fallback response template: {repr(response_template)}")
    return response_template


def create_sft_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Optional[Any],
    config: ExperimentConfig,
) -> Any:
    """
    Create an SFTTrainer configured for response-only training.

    Uses `unsloth`'s `train_on_responses_only` so the loss is only computed
    on the assistant response tokens, not the prompt/system tokens.

    Args:
        model: Model with LoRA adapters applied.
        tokenizer: Tokenizer.
        train_dataset: HF Dataset with 'text' column.
        eval_dataset: Optional HF Dataset with 'text' column.
        config: Full experiment config.

    Returns:
        Configured SFTTrainer instance (not yet trained).
    """
    from trl import SFTTrainer
    from transformers import TrainingArguments

    from src.train.callbacks import BestCheckpointCallback, WandbMetricsCallback
    from unsloth.chat_templates import train_on_responses_only

    tc = config.training
    output_dir = str(Path(config.paths.checkpoint_dir) / "sft")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        num_train_epochs=tc.num_train_epochs,
        max_steps=tc.max_steps,
        learning_rate=tc.learning_rate,
        warmup_ratio=tc.warmup_ratio,
        lr_scheduler_type=tc.lr_scheduler_type,
        weight_decay=tc.weight_decay,
        optim=tc.optim,
        logging_steps=tc.logging_steps,
        save_steps=tc.save_steps,
        eval_steps=tc.eval_steps if eval_dataset is not None else None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        bf16=tc.bf16,
        dataloader_num_workers=tc.dataloader_num_workers,
        report_to=tc.report_to,
        save_total_limit=tc.save_total_limit,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False if eval_dataset is not None else None,
        run_name=config.project.name,
        seed=config.project.seed,
    )


    # Callbacks
    callbacks = [WandbMetricsCallback()]
    if eval_dataset is not None:
        callbacks.append(BestCheckpointCallback(output_dir=output_dir))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=config.model.max_seq_length,
        dataset_text_field="text",
        packing=False,
        callbacks=callbacks,
    )
    
    # Veriry ChatML template existed
    # Unsloth map Nemotront o ChatML Template ( make sure <|im_start|> tokens exited)
    sample_text = train_dataset[0]["text"]
    if _RESPONSE_PART_ not in sample_text:
        raise ValueError(
            f"Response marker {_RESPONSE_PART_} not found in the first"
            f"training example. This means the chat template does not produce ChatML format"
        )
    
    trainer = train_on_responses_only(
        trainer=trainer,
        instruction_part=_INSTRUCTION_PART,
        response_part=_RESPONSE_PART_
    )

    return trainer


def run_sft(config_path: str, overrides: Optional[List[str]] = None) -> None:
    """
    Full SFT training pipeline.

    Steps:
        1. Load and resolve config
        2. Set random seeds
        3. Initialize wandb
        4. Load processed train/val data
        5. Load model and apply LoRA
        6. Build SFT datasets via formatter
        7. Create and run SFTTrainer
        8. Save best checkpoint / final adapter

    Args:
        config_path: Path to YAML experiment config.
        overrides: Optional list of "key.subkey=value" override strings.
    """
    import torch

    # 1. Load config
    config = load_config(config_path, overrides=overrides)
    config = resolve_paths(config)
    logger.info(f"Loaded config from {config_path}")

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
            name=config.project.name,
            config=config.model_dump(),
        )
        logger.info(f"wandb initialized: {config.project.wandb_project}")
    except ImportError:
        logger.warning("wandb not installed; skipping experiment tracking")

    # 4. Load processed data
    processed_dir = Path(config.paths.processed_data_dir)
    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"

    if not train_csv.exists():
        raise FileNotFoundError(
            f"Processed training data not found at {train_csv}. "
            f"Run scripts/prepare_data.py first."
        )

    train_df = pd.read_csv(train_csv)
    logger.info(f"Loaded {len(train_df)} training examples from {train_csv}")

    eval_df = None
    if val_csv.exists() and val_csv.stat().st_size > 0:
        eval_df = pd.read_csv(val_csv)
        logger.info(f"Loaded {len(eval_df)} validation examples from {val_csv}")
    else:
        logger.info("No validation data found; training without eval")

    # 5. Load model + apply LoRA
    from src.train.model import load_model_and_tokenizer, apply_lora

    model, tokenizer = load_model_and_tokenizer(config.model)
    model = apply_lora(model, config.lora)

    # 6. Build SFT datasets
    train_dataset = build_sft_dataset(
        train_df,
        tokenizer,
        reasoning_ratio=config.data.reasoning_ratio,
        seed=seed,
    )
    logger.info(f"Built SFT train dataset: {len(train_dataset)} examples")

    eval_dataset = None
    if eval_df is not None:
        eval_dataset = build_sft_dataset(
            eval_df,
            tokenizer,
            reasoning_ratio=config.data.reasoning_ratio,
            seed=seed,
        )
        logger.info(f"Built SFT eval dataset: {len(eval_dataset)} examples")

    # 7. Create and run trainer
    trainer = create_sft_trainer(
        model, tokenizer, train_dataset, eval_dataset, config
    )

    logger.info("Starting SFT training...")
    train_result = trainer.train()
    logger.info(f"Training complete. Metrics: {train_result.metrics}")

    # 8. Save final adapter
    final_dir = Path(config.paths.checkpoint_dir) / "sft" / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Saved final adapter to {final_dir}")

    # Log final metrics to wandb
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(train_result.metrics)
            wandb.finish()
            logger.info("wandb run finished")
    except ImportError:
        pass

    logger.info("SFT pipeline complete.")
