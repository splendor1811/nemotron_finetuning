"""Model loading, LoRA application, and merging utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

from src.config import LoraConfig, ModelConfig

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(config: ModelConfig) -> Tuple[Any, Any]:
    """
    Load model and tokenizer via Unsloth FastLanguageModel.

    Args:
        config: ModelConfig with model name, quantization settings, etc.

    Returns:
        (model, tokenizer) tuple.
    """
    from unsloth import FastLanguageModel

    logger.info(f"Loading model: {config.name}")
    logger.info(
        f"  max_seq_length={config.max_seq_length}, "
        f"load_in_4bit={config.load_in_4bit}"
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        trust_remote_code=config.trust_remote_code,
    )

    logger.info(f"Model loaded successfully. Dtype: {model.dtype}")
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

    return model, tokenizer


def apply_lora(model: Any, config: LoraConfig) -> Any:
    """
    Apply LoRA adapters via Unsloth FastLanguageModel.get_peft_model().

    Targets include in_proj/out_proj for Mamba-2 layers.
    Router (MoE gate) layers are excluded by default -- Unsloth handles this
    automatically by skipping non-linear layers.

    Args:
        model: Base model returned by load_model_and_tokenizer.
        config: LoraConfig with rank, alpha, target modules, etc.

    Returns:
        Model with LoRA adapters applied.
    """
    from unsloth import FastLanguageModel

    logger.info(f"Applying LoRA: r={config.r}, alpha={config.lora_alpha}")
    logger.info(f"  Target modules: {config.target_modules}")
    logger.info(f"  Gradient checkpointing: {config.use_gradient_checkpointing}")
    logger.info(f"  RSLoRA: {config.use_rslora}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias=config.bias,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        use_rslora=config.use_rslora,
        random_state=3407,
    )

    # Log trainable parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total > 0 else 0.0
    logger.info(
        f"LoRA applied: {trainable:,} trainable / {total:,} total params ({pct:.2f}%)"
    )

    return model


def merge_and_save(model: Any, tokenizer: Any, output_path: str) -> None:
    """
    Merge LoRA weights into the base model and save the merged result.

    Args:
        model: Model with LoRA adapters (from apply_lora).
        tokenizer: Tokenizer to save alongside the model.
        output_path: Directory to save the merged model and tokenizer.
    """
    from unsloth import FastLanguageModel

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Merging LoRA weights and saving to {output_dir}")

    # Save merged 16-bit model
    model.save_pretrained_merged(
        str(output_dir),
        tokenizer,
        save_method="merged_16bit",
    )

    logger.info(f"Merged model saved to {output_dir}")
