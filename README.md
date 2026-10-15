# Alignment Playground

Pure-Python SFT -> DPO pipeline on a single RTX 4060 8GB. Trains a real
base model through two stages so you can compare "before" (SFT-only) vs
"after" (SFT+DPO) behavior on the same prompts.

- Base model: `Qwen/Qwen2.5-3B` (QLoRA, 4-bit)
- SFT dataset: `teknium/OpenHermes-2.5` (60k subset)
- DPO dataset: `HuggingFaceH4/ultrafeedback_binarized` (18k subset)

## Setup

```bash
conda env create -f environment.yml
conda activate alignment-playground

# check your driver's CUDA version matches the torch wheel in environment.yml
nvidia-smi
```

## Pipeline

```bash
# 1. Pull + format both datasets
python data/prepare_data.py --stage all

# 2. Stage 1: SFT (base model -> checkpoints/sft)
python -m src.training.train_sft

# 3. Stage 2: DPO (checkpoints/sft -> checkpoints/dpo)
python -m src.training.train_dpo

# 4. Compare before vs after on the same prompts
python -m src.eval.compare
```

## What each script does

- `data/prepare_data.py` — downloads OpenHermes-2.5 + UltraFeedback, writes jsonl for both stages
- `src/training/train_sft.py` — trains a LoRA adapter on the 4-bit base model (SFTTrainer)
- `src/training/train_dpo.py` — loads the SFT checkpoint as both policy and frozen reference, trains a DPO adapter (DPOTrainer)
- `src/eval/compare.py` — loads both checkpoints, generates answers on a fixed prompt set, scores each pair with the DPO model acting as judge, writes `outputs/eval_reports/before_after_report.jsonl` and prints a summary table

The judge is fully local (no API keys) — it reuses the trained DPO model to
score its own answers against the SFT model's. This has some self-bias built
in; it's a proxy, not a neutral third party. Swap in a separate judge model
later if you want a cleaner signal.

## If you hit CUDA OOM

In order of preference, edit these before changing model size:

1. `configs/model_config.yaml` -> lower `max_seq_length` (512 -> 384 -> 256)
2. `configs/dpo_config.yaml` -> raise `gradient_accumulation_steps` further
3. `configs/model_config.yaml` -> switch `base_model` to `Qwen/Qwen2.5-1.5B`
   (everything else in the pipeline is unaffected by this change)

## Structure

See `configs/` for all hyperparameters — nothing is hardcoded in `src/`.
Training scripts are in `src/training/`, evaluation in `src/eval/`.
