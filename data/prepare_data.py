"""
Prepare the SFT and DPO datasets.

Writes the four files the training configs expect:

    data/processed/sft_train.jsonl   {"instruction": ..., "response": ...}
    data/processed/sft_val.jsonl
    data/processed/dpo_train.jsonl   {"prompt": ..., "chosen": ..., "rejected": ...}
    data/processed/dpo_val.jsonl

Sources:
    SFT: teknium/OpenHermes-2.5                      (streamed, no full download)
    DPO: HuggingFaceH4/ultrafeedback_binarized       (train_prefs / test_prefs)

Usage (from the project root):

    python data/prepare_data.py --stage all
    python data/prepare_data.py --stage sft
    python data/prepare_data.py --stage dpo

NOTE ON SIZES: the default sizes below were inferred from the training
logs (SFT: 2 epochs x 1250 steps x batch 8 = 10,000 examples; DPO:
290 steps x batch 16 = ~4,640 examples). Use the --*-train / --*-val
flags to change them.
"""

import argparse
import json
import os
import random
from typing import Dict, Iterable, List, Optional


OUTPUT_DIR = "data/processed"

SFT_DATASET = "teknium/OpenHermes-2.5"
DPO_DATASET = "HuggingFaceH4/ultrafeedback_binarized"


# ============================================================
# RECORD CONVERSION  (pure functions, no network needed)
# ============================================================

def convert_openhermes(record: dict, max_chars: int) -> Optional[Dict[str, str]]:
    """
    Convert one OpenHermes-2.5 record into {"instruction", "response"}.

    OpenHermes stores a "conversations" list of {"from", "value"} turns
    where "from" is "system", "human" or "gpt".

    Only single-turn conversations (optionally preceded by a system
    message) are kept, because the SFT trainer formats exactly one
    user turn and one assistant turn.
    """

    conversations = record.get("conversations") or []

    turns = [t for t in conversations if t.get("from") != "system"]

    if len(turns) != 2:
        return None

    if turns[0].get("from") != "human" or turns[1].get("from") != "gpt":
        return None

    instruction = str(turns[0].get("value", "")).strip()
    response = str(turns[1].get("value", "")).strip()

    if not instruction or not response:
        return None

    # Keep examples that fit comfortably inside max_seq_length so the
    # response (and its end token) are not truncated away.
    if len(instruction) + len(response) > max_chars:
        return None

    return {"instruction": instruction, "response": response}


def _last_assistant_message(messages: List[dict]) -> str:
    """
    Return the content of the last assistant message in a chat list.
    """

    for message in reversed(messages or []):

        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()

    return ""


def convert_ultrafeedback(record: dict, max_chars: int) -> Optional[Dict[str, str]]:
    """
    Convert one ultrafeedback_binarized record into
    {"prompt", "chosen", "rejected"}.

    In this dataset "chosen" and "rejected" are chat lists
    ([user, assistant]); the final assistant message is the response.
    """

    prompt = str(record.get("prompt", "")).strip()

    chosen = _last_assistant_message(record.get("chosen"))
    rejected = _last_assistant_message(record.get("rejected"))

    if not prompt or not chosen or not rejected:
        return None

    # A pair with identical text carries no preference signal.
    if chosen == rejected:
        return None

    if len(prompt) > max_chars:
        return None

    if len(prompt) + max(len(chosen), len(rejected)) > max_chars:
        return None

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


# ============================================================
# IO
# ============================================================

def _write_jsonl(path: str, records: Iterable[dict]) -> int:

    os.makedirs(os.path.dirname(path), exist_ok=True)

    count = 0

    with open(path, "w", encoding="utf-8") as f:

        for record in records:

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            count += 1

    return count


# ============================================================
# STAGES
# ============================================================

def prepare_sft(n_train: int, n_val: int, max_chars: int, seed: int):

    # Imported here so the converters above can be tested without
    # the `datasets` package installed.
    from datasets import load_dataset

    print(f"[SFT] Streaming {SFT_DATASET} ...")

    stream = load_dataset(SFT_DATASET, split="train", streaming=True)

    # Approximate shuffle: streaming datasets shuffle inside a buffer.
    stream = stream.shuffle(seed=seed, buffer_size=20_000)

    needed = n_train + n_val

    collected: List[Dict[str, str]] = []

    for record in stream:

        converted = convert_openhermes(record, max_chars)

        if converted is not None:
            collected.append(converted)

        if len(collected) >= needed:
            break

    if len(collected) < needed:
        print(
            f"[SFT] WARNING: only {len(collected)} usable examples found "
            f"(wanted {needed})."
        )

    train = collected[:n_train]
    val = collected[n_train:n_train + n_val]

    n1 = _write_jsonl(os.path.join(OUTPUT_DIR, "sft_train.jsonl"), train)
    n2 = _write_jsonl(os.path.join(OUTPUT_DIR, "sft_val.jsonl"), val)

    print(f"[SFT] wrote {n1:,} train and {n2:,} val examples to {OUTPUT_DIR}/")


def prepare_dpo(n_train: int, n_val: int, max_chars: int, seed: int):

    from datasets import load_dataset

    print(f"[DPO] Loading {DPO_DATASET} ...")

    train_raw = load_dataset(DPO_DATASET, split="train_prefs")
    val_raw = load_dataset(DPO_DATASET, split="test_prefs")

    train_raw = train_raw.shuffle(seed=seed)
    val_raw = val_raw.shuffle(seed=seed)

    def collect(dataset, limit: int) -> List[Dict[str, str]]:

        out: List[Dict[str, str]] = []

        for record in dataset:

            converted = convert_ultrafeedback(record, max_chars)

            if converted is not None:
                out.append(converted)

            if len(out) >= limit:
                break

        return out

    train = collect(train_raw, n_train)
    val = collect(val_raw, n_val)

    if len(train) < n_train:
        print(f"[DPO] WARNING: only {len(train)} train pairs found (wanted {n_train}).")

    if len(val) < n_val:
        print(f"[DPO] WARNING: only {len(val)} val pairs found (wanted {n_val}).")

    n1 = _write_jsonl(os.path.join(OUTPUT_DIR, "dpo_train.jsonl"), train)
    n2 = _write_jsonl(os.path.join(OUTPUT_DIR, "dpo_val.jsonl"), val)

    print(f"[DPO] wrote {n1:,} train and {n2:,} val pairs to {OUTPUT_DIR}/")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(description="Prepare SFT and DPO data")

    parser.add_argument("--stage", choices=["sft", "dpo", "all"], default="all")

    parser.add_argument("--sft-train", type=int, default=10_000)
    parser.add_argument("--sft-val", type=int, default=500)
    parser.add_argument("--dpo-train", type=int, default=4_640)
    parser.add_argument("--dpo-val", type=int, default=500)

    parser.add_argument(
        "--max-chars",
        type=int,
        default=1_800,
        help="Skip examples longer than this many characters "
             "(roughly 450 tokens, so they fit max_seq_length=512).",
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)

    if args.stage in ("sft", "all"):
        prepare_sft(args.sft_train, args.sft_val, args.max_chars, args.seed)

    if args.stage in ("dpo", "all"):
        prepare_dpo(args.dpo_train, args.dpo_val, args.max_chars, args.seed)


if __name__ == "__main__":
    main()