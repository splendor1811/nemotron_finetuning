# TODO — NVIDIA Nemotron Reasoning Challenge

## Phase 1: Foundation ✅
- [x] Project directory structure
- [x] src/config.py — Pydantic config loader with YAML inheritance
- [x] configs/base.yaml — Shared defaults
- [x] configs/experiments/ — LoRA r16/r32/r64
- [x] configs/rlvr/ — GRPO base/conservative
- [x] pyproject.toml (managed by uv)
- [x] requirements.txt
- [x] .gitignore
- [x] README.md

## Phase 2: Data Pipeline ✅
- [x] src/data/categories.py — Category specs + detection + matching
- [x] src/data/loader.py — CSV loading + stratified splitting
- [x] src/data/formatter.py — Chat template formatting (SFT + GRPO)
- [x] scripts/prepare_data.py — Entry point (tested, all 9500 rows categorized)

## Phase 3: SFT Training ✅
- [x] src/train/model.py — Unsloth + LoRA setup
- [x] src/train/sft.py — SFTTrainer wrapper with wandb
- [x] src/train/callbacks.py — WandbMetrics + BestCheckpoint callbacks
- [x] scripts/train_sft.py — Entry point

## Phase 4: Inference + Evaluation ✅
- [x] src/inference/answer_extract.py — <think> parsing + category normalization
- [x] src/inference/majority_vote.py — maj@K with numeric clustering
- [x] src/inference/engine.py — vLLM batch engine
- [x] src/eval/scoring.py — Category-aware matching
- [x] src/eval/metrics.py — Aggregated metrics + wandb logging
- [x] scripts/evaluate.py — Entry point
- [x] scripts/inference.py — Submission generation

## Phase 5: Kaggle Submission ✅
- [x] notebooks/submission.ipynb — G4 96GB GPU, BF16 vLLM, maj@64

## Phase 6: Synthetic Data ✅
- [x] src/data/synthetic.py — 6 category generators (tested)
- [x] scripts/generate_synthetic.py — Entry point

## Phase 7: RLVR Post-Training ✅
- [x] src/train/rewards.py — Correctness + format + reasoning quality rewards (tested)
- [x] src/train/rlvr.py — GRPO trainer
- [x] scripts/train_rlvr.py — Entry point
- [x] configs/rlvr/grpo_base.yaml + grpo_conservative.yaml

## Phase 8: SFT Pipeline Robustness Fix ✅
- [x] Remove `_get_response_template()` dead code
- [x] Remove `DataCollatorForCompletionOnlyLM` (removed from TRL 0.20, crashes on import)
- [x] Remove `data_collator` from SFTTrainer constructor
- [x] Add runtime verification that ChatML markers exist in training data
- [x] Extract template strings to module-level constants
- [x] Update docstring to reference `train_on_responses_only`
- [x] Clean up unused imports (`os`, `Dict`, `FastLanguageModel`)

## Next Steps (on training server)
- [ ] Run SFT training: `python scripts/train_sft.py --config configs/experiments/lora_r16.yaml`
- [ ] Evaluate baseline and SFT model
- [ ] Generate synthetic data and retrain
- [ ] Run RLVR post-training
- [ ] Submit to Kaggle and iterate