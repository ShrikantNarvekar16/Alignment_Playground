import torch
from peft import PeftModel

from src.utils.config import load_config
from src.models.load_model import (
    load_base_model,
    load_tokenizer,
)


MODEL_CONFIG = "configs/model_config.yaml"
SFT_CHECKPOINT = "checkpoints/sft_smoke"


def main():

    # ========================================================
    # CONFIG
    # ========================================================

    model_cfg = load_config(MODEL_CONFIG)

    # ========================================================
    # TOKENIZER
    # ========================================================

    print("Loading tokenizer...")

    tokenizer = load_tokenizer(
        model_cfg
    )

    # ========================================================
    # BASE MODEL
    # ========================================================

    print("Loading 4-bit base model...")

    model = load_base_model(
        model_cfg
    )

    # ========================================================
    # LOAD SFT LORA
    # ========================================================

    print("Loading SFT LoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        SFT_CHECKPOINT,
        is_trainable=False,
    )

    model.eval()

    # ========================================================
    # PROMPT
    # ========================================================

    prompt = "Explain what supervised fine-tuning is in simple terms."

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    # Use the same Qwen chat template that was used during SFT.
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    # Move inputs to the same device as the model.
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    # ========================================================
    # GENERATION
    # ========================================================

    print("\nGenerating response...\n")

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # ========================================================
    # DECODE ONLY NEW TOKENS
    # ========================================================

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    print("========================================")
    print("SFT MODEL RESPONSE")
    print("========================================")
    print(response)
    print("========================================")


if __name__ == "__main__":
    main()