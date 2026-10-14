import json

from datasets import Dataset


def _read_jsonl(path: str) -> list[dict]:
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

            if "instruction" not in record:
                raise ValueError(
                    f"Missing 'instruction' field at "
                    f"line {line_number} in {path}"
                )

            if "response" not in record:
                raise ValueError(
                    f"Missing 'response' field at "
                    f"line {line_number} in {path}"
                )

            instruction = str(
                record["instruction"]
            ).strip()

            response = str(
                record["response"]
            ).strip()

            if not instruction or not response:
                continue

            records.append(
                {
                    "instruction": instruction,
                    "response": response,
                }
            )

    return records


def load_sft_dataset(
    path: str,
    tokenizer=None,
) -> Dataset:

    records = _read_jsonl(path)

    if not records:
        raise ValueError(
            f"No valid records found in {path}"
        )

    formatted_records = []

    for record in records:

        instruction = record["instruction"]
        response = record["response"]

        # ----------------------------------------------------
        # QWEN CHAT TEMPLATE
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
                    "content": instruction,
                },
                {
                    "role": "assistant",
                    "content": response,
                },
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        # ----------------------------------------------------
        # FALLBACK FORMAT
        # ----------------------------------------------------

        else:

            text = (
                "### Instruction:\n"
                f"{instruction}\n\n"
                "### Response:\n"
                f"{response}"
            )

        formatted_records.append(
            {
                "text": text,
            }
        )

    dataset = Dataset.from_list(
        formatted_records
    )

    print(
        f"Loaded {len(dataset):,} SFT examples "
        f"from {path}"
    )

    return dataset