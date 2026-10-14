"""
Stage 2: Direct Preference Optimization.

Starts from the trained SFT LoRA adapter and continues training
that adapter with DPO preference data.

The policy is the SFT PeftModel (trainable). With ref_model=None, TRL
snapshots the starting adapter weights as a frozen "ref" adapter on the
same base model, so the reference is the SFT model (not the raw base
model) and no second Qwen model has to be loaded. This matters on an
RTX 4060 8GB GPU.

Usage:
    python -m src.training.train_dpo
"""

from trl import DPOTrainer, DPOConfig

from src.data.dpo_dataset import load_dpo_dataset
from src.models.load_model import (
    load_tokenizer,
    load_sft_model_for_dpo,
)
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed


logger = get_logger(__name__)


def main():

    # ========================================================
    # LOAD CONFIGURATION
    # ========================================================

    model_cfg = load_config(
        "configs/model_config.yaml"
    )

    train_cfg = load_config(
        "configs/dpo_config.yaml"
    )

    set_seed(
        train_cfg.get("seed", 42)
    )

    # ========================================================
    # TOKENIZER + DATASET
    # ========================================================

    logger.info(
        "Loading tokenizer and DPO datasets..."
    )

    tokenizer = load_tokenizer(
    model_cfg,
    padding_side="left",
    )

    train_dataset = load_dpo_dataset(
    train_cfg["dataset_path"],
    tokenizer=tokenizer,
    )

    eval_dataset = load_dpo_dataset(
    train_cfg["val_dataset_path"],
    tokenizer=tokenizer,
    )

    logger.info(
        f"Train examples: {len(train_dataset):,} | "
        f"Val examples: {len(eval_dataset):,}"
    )

    # ========================================================
    # LOAD SFT MODEL
    # ========================================================

    logger.info(
        "Loading SFT checkpoint as trainable DPO policy..."
    )

    policy_model = load_sft_model_for_dpo(
        model_cfg,
        train_cfg["sft_checkpoint"],
    )

    # ========================================================
    # DPO CONFIGURATION
    # ========================================================

    logger.info(
        "Creating DPO configuration..."
    )

    dpo_config = DPOConfig(

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        output_dir=train_cfg[
            "output_dir"
        ],

        # ----------------------------------------------------
        # DPO
        # ----------------------------------------------------

        beta=train_cfg[
            "beta"
        ],

        max_length=train_cfg[
            "max_seq_length"
        ],

        

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        num_train_epochs=train_cfg[
            "num_train_epochs"
        ],

        per_device_train_batch_size=train_cfg[
            "per_device_train_batch_size"
        ],

        per_device_eval_batch_size=train_cfg.get(
            "per_device_eval_batch_size",
            1,
        ),

        gradient_accumulation_steps=train_cfg[
            "gradient_accumulation_steps"
        ],

        gradient_checkpointing=train_cfg[
            "gradient_checkpointing"
        ],

        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        learning_rate=train_cfg[
            "learning_rate"
        ],

        lr_scheduler_type=train_cfg[
            "lr_scheduler_type"
        ],

        warmup_steps=train_cfg[
        "warmup_steps"
        ],

        optim=train_cfg[
            "optim"
        ],

        max_grad_norm=train_cfg[
            "max_grad_norm"
        ],

        # ----------------------------------------------------
        # LOGGING
        # ----------------------------------------------------

        logging_steps=train_cfg[
            "logging_steps"
        ],

        eval_strategy="steps",

        eval_steps=train_cfg[
            "eval_steps"
        ],

        save_steps=train_cfg[
            "save_steps"
        ],

        save_total_limit=train_cfg[
            "save_total_limit"
        ],

        report_to=train_cfg[
            "report_to"
        ],

        # ----------------------------------------------------
        # PRECISION
        # ----------------------------------------------------

        bf16=train_cfg.get(
            "bf16",
            True,
        ),

        # ----------------------------------------------------
        # REPRODUCIBILITY
        # ----------------------------------------------------

        seed=train_cfg.get(
            "seed",
            42,
        ),
    )

    # ========================================================
    # DPO TRAINER
    # ========================================================

    logger.info(
        "Creating DPOTrainer..."
    )

    trainer = DPOTrainer(

        # Existing SFT PeftModel.
        #
        # IMPORTANT:
        # Do NOT pass peft_config here.
        model=policy_model,

        # No second model is loaded.
        #
        # Because model is a PeftModel, TRL keeps a frozen copy of
        # the starting (SFT) adapter as the reference.
        ref_model=None,

        args=dpo_config,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,

        processing_class=tokenizer,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    logger.info(
        "Starting DPO training..."
    )

    trainer.train()

    # ========================================================
    # SAVE
    # ========================================================

    logger.info(
        f"Saving DPO adapter to "
        f"{train_cfg['output_dir']}"
    )

    trainer.save_model(
        train_cfg["output_dir"]
    )

    tokenizer.save_pretrained(
        train_cfg["output_dir"]
    )

    logger.info(
        "DPO stage complete."
    )


if __name__ == "__main__":
    main()