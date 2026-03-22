# Lessons Learned — NVIDIA Nemotron Reasoning Challenge

## Architecture
- Nemotron-3-Nano-30B-A3B is a hybrid Mamba-2 + Transformer MoE (52 layers: 23 Mamba, 23 MoE, 6 GQA)
- 30B total params, 3B active — sparse MoE means only 6 of 128 experts activate per token
- LoRA must target in_proj/out_proj for Mamba-2 layers, not just attention projections
- Do NOT fine-tune MoE router layers (Unsloth disables by default)
- Model uses <think>...</think> tokens for reasoning traces (token IDs 12, 13)
- attn_implementation must be "eager" for this hybrid architecture

## Data
- train.csv has ~69K rows across 6 balanced categories (~1,555-1,602 each)
- test.csv has 34 rows (placeholder; real test is hidden on Kaggle)
- train.csv only has final answers, no reasoning traces — must generate synthetic CoT

## Competition
- Evaluation: pass@1 with maj@64 (majority voting across 64 generations)
- Kaggle GPU: G4 with 96GB VRAM — can load full BF16 model
- All 6 categories have verifiable answers — perfect for RLVR

## Training
- Recommended inference: temperature=1.0, top_p=1.0 for reasoning (NVIDIA official)
- 75%+ reasoning examples in training data to preserve reasoning capability
- RLVR/GRPO improves pass@1 by +5-15% over SFT (OLMo 3 / DeepSeek evidence)

