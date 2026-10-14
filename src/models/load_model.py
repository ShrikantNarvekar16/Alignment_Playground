import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


# ============================================================
# BITSANDBYTES CONFIGURATION
# ============================================================

def _bnb_config(model_cfg: dict) -> BitsAndBytesConfig:
    """
    Create the 4-bit BitsAndBytes configuration used for QLoRA.
    """

    dtype_name = model_cfg.get(
        "bnb_4bit_compute_dtype",
        "bfloat16",
    )

    compute_dtype = getattr(torch, dtype_name)

    return BitsAndBytesConfig(
        load_in_4bit=model_cfg.get(
            "load_in_4bit",
            True,
        ),
        bnb_4bit_quant_type=model_cfg.get(
            "bnb_4bit_quant_type",
            "nf4",
        ),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=model_cfg.get(
            "bnb_4bit_use_double_quant",
            True,
        ),
    )


# ============================================================
# TOKENIZER
# ============================================================

def load_tokenizer(
    model_cfg: dict,
    padding_side: str = "right",
):
    """
    Load the tokenizer.

    SFT uses right padding.
    DPO uses left padding.
    """

    logger.info(
        f"Loading tokenizer: {model_cfg['tokenizer']}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["tokenizer"],
        trust_remote_code=model_cfg.get(
            "trust_remote_code",
            True,
        ),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = padding_side

    return tokenizer


# ============================================================
# BASE MODEL
# ============================================================

def load_base_model(model_cfg: dict):
    """
    Load the raw 4-bit quantized base model.

    No LoRA adapter is attached here.
    """

    logger.info(
        f"Loading base model: {model_cfg['base_model']}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        quantization_config=_bnb_config(model_cfg),
        device_map="auto",
        trust_remote_code=model_cfg.get(
            "trust_remote_code",
            True,
        ),
    )

    model.config.use_cache = False

    return model


# ============================================================
# NEW LORA
# ============================================================

def attach_new_lora(
    model,
    train_cfg: dict,
):
    """
    Prepare a quantized model for QLoRA training and
    attach a fresh LoRA adapter.

    Used during SFT.
    """

    logger.info(
        "Preparing model for k-bit training..."
    )

    model = prepare_model_for_kbit_training(model)

    logger.info(
        "Creating LoRA adapter..."
    )

    lora_config = LoraConfig(
        r=train_cfg["lora_r"],
        lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        target_modules=train_cfg["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    return model


# ============================================================
# LOAD SFT MODEL FOR DPO
# ============================================================

def load_sft_model_for_dpo(
    model_cfg: dict,
    sft_checkpoint: str,
):
    """
    Load the 4-bit base model and the trained SFT LoRA adapter.

    The returned PeftModel is trainable and becomes the DPO policy.

    IMPORTANT:
    We do NOT merge the adapter into the quantized base model.
    We also do NOT create another LoRA adapter here.

    TRL's DPOTrainer can continue training an existing PeftModel.
    """

    logger.info(
        "Loading base model for DPO..."
    )

    model = load_base_model(
        model_cfg
    )

    logger.info(
        "Preparing base model for k-bit training..."
    )

    model = prepare_model_for_kbit_training(
        model
    )

    logger.info(
        f"Loading SFT adapter from: {sft_checkpoint}"
    )

    model = PeftModel.from_pretrained(
        model,
        sft_checkpoint,
        is_trainable=True,
    )

    model.config.use_cache = False

    logger.info(
        "SFT adapter loaded as trainable DPO policy."
    )

    model.print_trainable_parameters()

    return model


# ============================================================
# LOAD CHECKPOINT FOR INFERENCE / CHAT
# ============================================================

def load_model_for_checkpoint(
    model_config: dict,
    checkpoint_path: str,
):
    """
    Load a trained LoRA checkpoint for inference/chat.

    The checkpoint is attached to the 4-bit base model.
    The adapter is not made trainable.
    """

    logger.info(
        "Loading base model for inference..."
    )

    model = load_base_model(
        model_config
    )

    logger.info(
        f"Loading LoRA checkpoint from: {checkpoint_path}"
    )

    model = PeftModel.from_pretrained(
        model,
        checkpoint_path,
        is_trainable=False,
    )

    model.config.use_cache = True

    model.eval()

    logger.info(
        "Checkpoint loaded successfully for inference."
    )

    return model
