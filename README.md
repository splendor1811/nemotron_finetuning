# NVIDIA Nemotron Reasoning Challenge Pipeline

Production-ready fine-tuning and inference pipeline for the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) on Kaggle.

## Overview

**Goal**: Maximize reasoning accuracy of **Nemotron-3-Nano-30B-A3B** on rule-inference tasks across 6 categories.

**Model**: NVIDIA-Nemotron-3-Nano-30B-A3B — Hybrid Mamba-2 + Transformer MoE architecture (30B total params, 3B active per token).

**Pipeline**: SFT with LoRA → Evaluation → RLVR with GRPO → Kaggle Submission

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐
│  Raw Data    │───▶│  SFT Train   │───▶│  RLVR/GRPO   │───▶│  Inference  │
│  (train.csv) │    │  (LoRA)      │    │  (rewards)   │    │  (vLLM)    │
└─────────────┘    └──────────────┘    └──────────────┘    └────────────┘
       │                                                          │
       ▼                                                          ▼
┌─────────────┐                                          ┌────────────┐
│  Synthetic   │                                          │ submission │
│  Data Gen    │                                          │   .csv     │
└─────────────┘                                          └────────────┘
```

### Task Categories

| Category | Answer Type | Matching |
|----------|-------------|----------|
| Bit Manipulation | 8-char binary strings | Exact match |
| Gravitational Constant | Decimal numbers | Approximate (tolerance) |
| Unit Conversion | Decimal numbers | Approximate (tolerance) |
| Text Encryption | English phrases | Case-insensitive exact |
| Numeral System | Roman numerals | Exact match |
| Equation Transformation | Symbolic strings | Exact match |

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA 12.x
- [uv](https://docs.astral.sh/uv/) for package management
- GPU: 60GB+ VRAM for BF16 training (H100/A100), or 24GB+ for QLoRA (4-bit)
- Kaggle: G4 GPU with 96GB VRAM for submission

### Installation

```bash
# Create virtual environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# For Unsloth (optional, for faster training)
uv pip install -e ".[unsloth]"
```

### Data Setup

Data should be symlinked at `data/raw/`:
```bash
ls data/raw/
# train.csv  test.csv
```

### wandb Setup

```bash
wandb login
# Optionally set your entity in configs/base.yaml → project.wandb_entity
```

---

## Pipeline Stages

### 1. Data Preparation

```bash
python scripts/prepare_data.py --config configs/base.yaml
```

- Loads `train.csv`, auto-detects 6 categories from prompt content
- Creates stratified train/val split (default 90/10)
- Formats into chat template with synthetic reasoning traces
- Saves as HuggingFace Dataset in `data/processed/`

### 2. SFT Fine-Tuning

```bash
python scripts/train_sft.py --config configs/experiments/lora_r16.yaml
```

- Loads Nemotron-3-Nano-30B via Unsloth
- Applies LoRA to all linear projections (including Mamba-2 in_proj/out_proj)
- Trains with SFTTrainer, masking system/user tokens
- Logs to wandb in real-time

**Key config options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora.r` | 16 | LoRA rank (16/32/64) |
| `lora.lora_alpha` | 32 | LoRA scaling (typically 2x r) |
| `training.learning_rate` | 5e-5 | Peak learning rate |
| `training.num_train_epochs` | 3 | Training epochs |
| `training.per_device_train_batch_size` | 2 | Batch size per GPU |
| `data.reasoning_ratio` | 0.8 | Fraction with reasoning traces |

### 3. Evaluation

```bash
python scripts/evaluate.py --config configs/base.yaml \
    --model-path outputs/checkpoints/sft-best
```

- Runs inference on validation set
- Computes per-category accuracy, pass@1, maj@K
- Logs detailed metrics and sample predictions to wandb

### 4. RLVR Post-Training (After SFT)

```bash
python scripts/train_rlvr.py --config configs/rlvr/grpo_base.yaml \
    --sft-checkpoint outputs/checkpoints/sft-best
```

- Loads SFT checkpoint as starting policy
- Trains with GRPO using verifiable reward functions:
  - **Correctness** (weight 3.0): Is the answer right?
  - **Format** (weight 1.0): Proper `<think>...</think>` structure?
  - **Reasoning quality** (weight 0.5): Structured step-by-step thinking?
- Progressive multi-stage training (original data -> synthetic -> harder problems)

**Key monitoring** (these are normal for GRPO):
- Loss **increases** (measures KL divergence from initial policy)
- Reward should increase
- reward_std should stay > 0.1 (no mode collapse)

### 5. Synthetic Data Generation

```bash
python scripts/generate_synthetic.py --config configs/base.yaml \
    --n-per-category 500
```

- Generates new rule-inference problems per category
- Produces prompt + answer + reasoning trace
- Saves to `data/synthetic/` for augmented training

### 6. Inference & Submission

```bash
python scripts/inference.py --config configs/base.yaml \
    --input data/raw/test.csv \
    --output outputs/submissions/submission.csv
```

- Batch inference via vLLM with maj@64 majority voting
- Category-aware answer extraction and matching
- Outputs `submission.csv` for Kaggle upload

---

## Configuration System

### YAML Inheritance

Configs support `_base_` inheritance — child configs override parent values:

```yaml
# configs/experiments/lora_r32.yaml
_base_: ../base.yaml    # Inherits everything from base.yaml

lora:
  r: 32                 # Override just this field
  lora_alpha: 64

training:
  learning_rate: 3.0e-5
```

### CLI Overrides

Override any config value from the command line:

```bash
python scripts/train_sft.py --config configs/base.yaml \
    --override lora.r=64 \
    --override training.learning_rate=1e-4
```

### Available Configs

| Config | Purpose |
|--------|---------|
| `configs/base.yaml` | Shared defaults for all experiments |
| `configs/experiments/lora_r16.yaml` | LoRA rank 16 (baseline) |
| `configs/experiments/lora_r32.yaml` | LoRA rank 32 (more capacity) |
| `configs/experiments/lora_r64.yaml` | LoRA rank 64 (maximum capacity) |
| `configs/rlvr/grpo_base.yaml` | GRPO with standard settings |
| `configs/rlvr/grpo_conservative.yaml` | GRPO with lower LR, higher KL penalty |
| `configs/inference_kaggle.yaml` | Kaggle G4 GPU inference settings |

---

## Project Structure

```
kaggle_nvidia_nemotron/
├── configs/                       # YAML experiment configs
│   ├── base.yaml                  # Shared defaults
│   ├── experiments/               # SFT experiment variants
│   ├── rlvr/                      # GRPO/RLVR configs
│   └── inference_kaggle.yaml      # Kaggle submission config
├── src/
│   ├── config.py                  # Pydantic config with YAML inheritance
│   ├── data/
│   │   ├── categories.py          # Category specs, detection, matching
│   │   ├── loader.py              # CSV loading, splitting
│   │   ├── formatter.py           # Chat template formatting
│   │   └── synthetic.py           # Synthetic data generators
│   ├── train/
│   │   ├── model.py               # Unsloth model + LoRA setup
│   │   ├── sft.py                 # SFT trainer
│   │   ├── rlvr.py                # GRPO/RLVR trainer
│   │   ├── rewards.py             # Verifiable reward functions
│   │   └── callbacks.py           # Training callbacks
│   ├── inference/
│   │   ├── engine.py              # vLLM batch inference
│   │   ├── answer_extract.py      # Parse model output
│   │   └── majority_vote.py       # maj@K voting
│   └── eval/
│       ├── scoring.py             # Category-aware matching
│       └── metrics.py             # Aggregated metrics + wandb
├── scripts/                       # CLI entry points
├── notebooks/                     # Kaggle submission notebook
├── tasks/                         # Progress tracking
├── data/                          # Data (gitignored)
├── outputs/                       # Checkpoints & submissions (gitignored)
├── pyproject.toml                 # Dependencies (managed by uv)
└── requirements.txt               # Flat deps for Kaggle notebook
```

---

## Experiment Tracking (wandb)

### Key Metrics to Monitor

**SFT Training:**
- `train/loss` — should decrease
- `eval/loss` — should decrease (watch for overfitting gap)
- `eval/accuracy_per_category` — per-category breakdown

**RLVR Training:**
- `reward` — should increase
- `reward_std` — should stay > 0.1 (no mode collapse)
- `kl` — should grow moderately (< 0.5)
- `train/loss` — **increases** (this is normal for GRPO)

---

## Troubleshooting

### Common Issues

**OOM during training:**
- Reduce `training.per_device_train_batch_size` to 1
- Enable `model.load_in_4bit: true` for QLoRA
- Reduce `data.max_seq_length` to 2048

**Mamba layer LoRA errors:**
- Ensure `unsloth>=2024.12` (Mamba-2 support)
- Remove `in_proj`/`out_proj` from `lora.target_modules` as fallback
- Set `model.trust_remote_code: true`

**vLLM inference failures:**
- Ensure `vllm>=0.7.0` with Nemotron support
- Set `--trust-remote-code` flag
- Use `--mamba-ssm-cache-dtype float32` for accuracy

**RLVR mode collapse (reward_std -> 0):**
- Increase `rlvr.num_generations` (32 or 64)
- Increase `rlvr.temperature` (0.8-1.0)
- Reduce `rlvr.learning_rate` by half

### GPU Memory Requirements

| Configuration | VRAM Required |
|---------------|---------------|
| BF16 inference | ~60 GB |
| BF16 LoRA training | ~65 GB |
| QLoRA (4-bit) training | ~24 GB |
| FP8 inference | ~30 GB |
| Kaggle G4 (96GB) | Full BF16 OK |
