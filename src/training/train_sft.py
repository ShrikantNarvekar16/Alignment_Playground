"""
Stage 1: Supervised Fine-Tuning.

Trains a LoRA adapter on top of a 4-bit base model using
instruction/response pairs.

Usage:
    python -m src.training.train_sft
"""

from trl import SFTTrainer, SFTConfig

from src.data.sft_dataset import load_sft_dataset
from src.models.load_model import (
    load_base_model,
    load_tokenizer,
    attach_new_lora,
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
        "configs/sft_config.yaml"
    )

    # ========================================================
    # RANDOM SEED
    # ========================================================

    seed = train_cfg.get(
        "seed",
        42,
    )

    set_seed(seed)

    # ========================================================
    # TOKENIZER
    # ========================================================

    logger.info(
        "Loading tokenizer..."
    )

    tokenizer = load_tokenizer(
        model_cfg,
        padding_side="right",
    )

    # ========================================================
    # DATASETS
    # ========================================================

    logger.info(
        "Loading training dataset..."
    )

    train_dataset = load_sft_dataset(
        train_cfg["dataset_path"],
        tokenizer=tokenizer,
    )

    logger.info(
        "Loading validation dataset..."
    )

    eval_dataset = load_sft_dataset(
        train_cfg["val_dataset_path"],
        tokenizer=tokenizer,
    )

    logger.info(
        f"Train examples: {len(train_dataset):,} | "
        f"Validation examples: {len(eval_dataset):,}"
    )

    # ========================================================
    # BASE MODEL
    # ========================================================

    logger.info(
        "Loading 4-bit base model..."
    )

    model = load_base_model(
        model_cfg
    )

    # ========================================================
    # LoRA
    # ========================================================

    logger.info(
        "Attaching LoRA adapter..."
    )

    model = attach_new_lora(
        model,
        train_cfg,
    )

    # ========================================================
    # GRADIENT CHECKPOINTING
    # ========================================================

    if train_cfg.get(
        "gradient_checkpointing",
        True,
    ):

        logger.info(
            "Gradient checkpointing enabled."
        )

        model.config.use_cache = False

    # ========================================================
    # SFT CONFIG
    # ========================================================

    logger.info(
        "Creating SFT configuration..."
    )

    sft_config = SFTConfig(

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        output_dir=train_cfg[
            "output_dir"
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

        gradient_checkpointing=train_cfg.get(
            "gradient_checkpointing",
            True,
        ),

        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        # ----------------------------------------------------
        # OPTIMIZATION
        # ----------------------------------------------------

        learning_rate=train_cfg[
            "learning_rate"
        ],

        lr_scheduler_type=train_cfg[
            "lr_scheduler_type"
        ],

        warmup_steps=train_cfg.get(
            "warmup_steps",
            0,
        ),

        weight_decay=train_cfg[
            "weight_decay"
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

        report_to=train_cfg.get(
            "report_to",
            "none",
        ),

        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        eval_strategy="steps",

        eval_steps=train_cfg[
            "eval_steps"
        ],

        # ----------------------------------------------------
        # CHECKPOINTS
        # ----------------------------------------------------

        save_steps=train_cfg[
            "save_steps"
        ],

        save_total_limit=train_cfg[
            "save_total_limit"
        ],

        # ----------------------------------------------------
        # PRECISION
        # ----------------------------------------------------

        bf16=train_cfg.get(
            "bf16",
            True,
        ),

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        dataset_text_field="text",

        max_length=model_cfg[
            "max_seq_length"
        ],

        # ----------------------------------------------------
        # REPRODUCIBILITY
        # ----------------------------------------------------

        seed=seed,
    )

    # ========================================================
    # TRAINER
    # ========================================================

    logger.info(
        "Creating SFTTrainer..."
    )

    trainer = SFTTrainer(

        model=model,

        args=sft_config,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,

        processing_class=tokenizer,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    logger.info(
        "Starting SFT training..."
    )

    trainer.train()

    # ========================================================
    # SAVE
    # ========================================================

    logger.info(
        f"Saving SFT adapter to "
        f"{train_cfg['output_dir']}"
    )

    trainer.save_model(
        train_cfg["output_dir"]
    )

    tokenizer.save_pretrained(
        train_cfg["output_dir"]
    )

    logger.info(
        "SFT stage complete."
    )


if __name__ == "__main__":
    main()