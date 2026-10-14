import json

from datasets import Dataset


# ============================================================
# JSONL LOADING
# ============================================================

def _read_jsonl(path: str) -> list[dict]:
    """
    Read DPO JSONL records.

    Expected format:

        {
            "prompt": "...",
            "chosen": "...",
            "rejected": "..."
        }
    """

    records = []

    with open(path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"Invalid JSON at line {line_number} "
                    f"in {path}"
                ) from error

            # ------------------------------------------------
            # VALIDATE REQUIRED FIELDS
            # ------------------------------------------------

            for field in (
                "prompt",
                "chosen",
                "rejected",
            ):

                if field not in record:

                    raise ValueError(
                        f"Missing '{field}' field at "
                        f"line {line_number} in {path}"
                    )

            prompt = str(
                record["prompt"]
            ).strip()

            chosen = str(
                record["chosen"]
            ).strip()

            rejected = str(
                record["rejected"]
            ).strip()

            # ------------------------------------------------
            # SKIP EMPTY RECORDS
            # ------------------------------------------------

            if (
                not prompt
                or not chosen
                or not rejected
            ):
                continue

            records.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

    return records


# ============================================================
# DATASET LOADER
# ============================================================

def load_dpo_dataset(
    path: str,
    tokenizer=None,
) -> Dataset:
    """
    Load a DPO dataset.

    Output columns:

        prompt
        chosen
        rejected

    When a tokenizer with apply_chat_template() is provided,
    the prompt is formatted using the model's native chat
    template.

    The chosen/rejected responses remain separate because
    DPOTrainer needs to compare them.
    """

    records = _read_jsonl(path)

    if not records:

        raise ValueError(
            f"No valid DPO records found in {path}"
        )

    formatted_records = []

    # ========================================================
    # FORMAT RECORDS
    # ========================================================

    for record in records:

        prompt = record["prompt"]
        chosen = record["chosen"]
        rejected = record["rejected"]

        # ----------------------------------------------------
        # MODEL CHAT TEMPLATE
        # ----------------------------------------------------

        if (
            tokenizer is not None
            and hasattr(
                tokenizer,
                "apply_chat_template",
            )
        ):

            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        else:

            formatted_prompt = (
                "### Instruction:\n"
                f"{prompt}\n\n"
                "### Response:\n"
            )

        formatted_records.append(
            {
                "prompt": formatted_prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    dataset = Dataset.from_list(
        formatted_records
    )

    print(
        f"Loaded {len(dataset):,} DPO examples "
        f"from {path}"
    )

    return dataset