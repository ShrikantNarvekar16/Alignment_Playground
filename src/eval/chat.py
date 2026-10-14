"""
Interactive SFT vs DPO comparison.

Run from project root:

    python -m src.eval.chat

Enter your own prompt and the script will generate:
    1. SFT answer
    2. DPO answer

The 4-bit base model is loaded ONCE and both LoRA adapters are attached
to it, so switching between SFT and DPO is instant (no model reloads).
"""

import logging

from peft import PeftModel

from src.eval.generate_answers import generate
from src.models.load_model import load_base_model, load_tokenizer
from src.utils.config import load_yaml_config


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

MODEL_CONFIG_PATH = "configs/model_config.yaml"

SFT_CHECKPOINT = "checkpoints/sft"
DPO_CHECKPOINT = "checkpoints/dpo"

MAX_NEW_TOKENS = 256

TEMPERATURE = 0.7
TOP_P = 0.9


# ============================================================
# CLEAN GENERATED ANSWER
# ============================================================

def clean_answer(text: str) -> str:
    """
    Light cleanup of a generated answer.

    generate() already stops at <|im_end|> and strips special tokens,
    so this only trims whitespace and removes a stray template header
    if the model happens to emit one.
    """

    if not text:
        return ""

    text = text.strip()

    if text.startswith("### Response:"):
        text = text[len("### Response:"):].strip()

    return text


# ============================================================
# LOAD BOTH ADAPTERS ON ONE BASE MODEL
# ============================================================

def load_model_with_both_adapters(model_config: dict):
    """
    Load the 4-bit base model once and attach both adapters.

    Adapter names:
        "sft"  -> checkpoints/sft
        "dpo"  -> checkpoints/dpo

    Use model.set_adapter("sft") / model.set_adapter("dpo") to switch.
    """

    logger.info("Loading 4-bit base model...")

    base = load_base_model(model_config)

    logger.info(f"Loading SFT adapter from: {SFT_CHECKPOINT}")

    model = PeftModel.from_pretrained(
        base,
        SFT_CHECKPOINT,
        adapter_name="sft",
        is_trainable=False,
    )

    logger.info(f"Loading DPO adapter from: {DPO_CHECKPOINT}")

    model.load_adapter(
        DPO_CHECKPOINT,
        adapter_name="dpo",
        is_trainable=False,
    )

    # load_base_model() disables the KV cache for training; turn it
    # back on for fast generation.
    model.config.use_cache = True

    model.eval()

    return model


# ============================================================
# GENERATE WITH A NAMED ADAPTER
# ============================================================

def answer_with_adapter(model, tokenizer, adapter_name: str, prompt: str) -> str:

    model.set_adapter(adapter_name)

    answer = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    return clean_answer(answer)


# ============================================================
# DISPLAY
# ============================================================

def print_comparison(
    prompt: str,
    sft_answer: str,
    dpo_answer: str,
):
    """
    Print clean SFT vs DPO comparison.
    """

    print()
    print("=" * 100)
    print("PROMPT")
    print("=" * 100)
    print(prompt)

    print()
    print("-" * 100)
    print("SFT")
    print("-" * 100)
    print(sft_answer)

    print()
    print("-" * 100)
    print("DPO")
    print("-" * 100)
    print(dpo_answer)

    print()
    print("=" * 100)
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    model_config = load_yaml_config(MODEL_CONFIG_PATH)

    logger.info("Loading tokenizer...")

    tokenizer = load_tokenizer(model_config, padding_side="left")

    model = load_model_with_both_adapters(model_config)

    print()
    print("=" * 100)
    print("LOCAL LLM — SFT vs DPO INTERACTIVE COMPARISON")
    print("=" * 100)
    print()
    print("Enter any prompt.")
    print("Both SFT and DPO will answer it.")
    print()
    print("Type 'exit', 'quit', or 'q' to stop.")
    print()

    while True:

        try:
            user_prompt = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_prompt:
            continue

        if user_prompt.lower() in {"exit", "quit", "q"}:
            print("Exiting...")
            break

        print()
        print("Generating SFT response...")

        sft_answer = answer_with_adapter(
            model, tokenizer, "sft", user_prompt
        )

        print("Generating DPO response...")

        dpo_answer = answer_with_adapter(
            model, tokenizer, "dpo", user_prompt
        )

        print_comparison(
            prompt=user_prompt,
            sft_answer=sft_answer,
            dpo_answer=dpo_answer,
        )


if __name__ == "__main__":
    main()