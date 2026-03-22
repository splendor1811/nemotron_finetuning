"""Configuration system with Pydantic validation and YAML inheritance."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "nemotron-reasoning"
    seed: int = 3407
    wandb_project: str = "nemotron-reasoning-challenge"
    wandb_entity: Optional[str] = None


class PathsConfig(BaseModel):
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    synthetic_data_dir: str = "data/synthetic"
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"

    def resolve(self, base_dir: Path) -> "PathsConfig":
        """Resolve relative paths against a base directory."""
        data = self.model_dump()
        for key, value in data.items():
            p = Path(value)
            if not p.is_absolute():
                data[key] = str(base_dir / p)
        return PathsConfig(**data)


class DataConfig(BaseModel):
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    val_fraction: float = Field(default=0.1, ge=0.0, le=0.5)
    val_strategy: str = "stratified"
    max_seq_length: int = 4096
    reasoning_ratio: float = Field(default=0.8, ge=0.0, le=1.0)


class ModelConfig(BaseModel):
    name: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    trust_remote_code: bool = True
    max_seq_length: int = 4096
    load_in_4bit: bool = False
    load_in_8bit: bool = False


class LoraConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: List[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "in_proj", "out_proj",
    ])
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    use_rslora: bool = False


class TrainingConfig(BaseModel):
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    max_steps: int = -1
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    bf16: bool = True
    dataloader_num_workers: int = 4
    report_to: str = "wandb"
    save_total_limit: int = 3


class RewardWeightsConfig(BaseModel):
    correctness: float = 3.0
    format: float = 1.0
    reasoning_quality: float = 0.5


class RlvrConfig(BaseModel):
    enabled: bool = False
    algorithm: str = "grpo"
    learning_rate: float = 1e-6
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_generations: int = 16
    max_prompt_length: int = 1024
    max_completion_length: int = 2048
    temperature: float = 0.7
    num_train_epochs: int = 2
    kl_coef: float = 0.001
    max_grad_norm: float = 0.5
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    report_to: str = "wandb"
    save_steps: int = 100
    logging_steps: int = 1
    reward_weights: RewardWeightsConfig = Field(default_factory=RewardWeightsConfig)


class InferenceConfig(BaseModel):
    engine: str = "vllm"
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 2048
    num_generations: int = 64
    batch_size: int = 8
    gpu_memory_utilization: float = 0.90


class ExperimentConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    rlvr: RlvrConfig = Field(default_factory=RlvrConfig)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml_with_inheritance(config_path: Path) -> Dict[str, Any]:
    """Load a YAML file, resolving _base_ inheritance."""
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    base_path = data.pop("_base_", None)
    if base_path is not None:
        base_file = (config_path.parent / base_path).resolve()
        base_data = _load_yaml_with_inheritance(base_file)
        data = _deep_merge(base_data, data)

    return data


def load_config(config_path: str, overrides: Optional[List[str]] = None) -> ExperimentConfig:
    """
    Load experiment config from YAML with inheritance and CLI overrides.

    Args:
        config_path: Path to YAML config file.
        overrides: List of "key.subkey=value" override strings.

    Returns:
        Validated ExperimentConfig.
    """
    path = Path(config_path).resolve()
    data = _load_yaml_with_inheritance(path)

    if overrides:
        for override in overrides:
            key, _, value = override.partition("=")
            keys = key.strip().split(".")
            target = data
            for k in keys[:-1]:
                target = target.setdefault(k, {})
            # Try to parse as number/bool
            v = value.strip()
            if v.lower() == "true":
                target[keys[-1]] = True
            elif v.lower() == "false":
                target[keys[-1]] = False
            elif v.lower() == "null" or v.lower() == "none":
                target[keys[-1]] = None
            else:
                try:
                    target[keys[-1]] = int(v)
                except ValueError:
                    try:
                        target[keys[-1]] = float(v)
                    except ValueError:
                        target[keys[-1]] = v

    return ExperimentConfig(**data)


def resolve_paths(config: ExperimentConfig, base_dir: Optional[str] = None) -> ExperimentConfig:
    """Resolve all relative paths in config against base_dir."""
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)
    config.paths = config.paths.resolve(base_dir)
    return config
