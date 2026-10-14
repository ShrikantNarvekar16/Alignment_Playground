"""
Generate answers from both SFT ("before") and DPO ("after") checkpoints.

The exact same evaluation prompts are used for both models, and the
prompts are formatted with the Qwen chat template -- the same format
used by SFT and DPO training.

Usage:
    python -m src.eval.generate_answers

Output:
    outputs/generations/before_after.jsonl
"""

import gc
import json
import os
import re
from typing import List, Optional

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from src.models.load_model import load_model_for_checkpoint, load_tokenizer
from src.utils.config import load_config
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Generation settings
# ---------------------------------------------------------------------------

# Room for a full answer. 300 tokens cut most answers off mid-sentence.
MAX_NEW_TOKENS = 450

# Mild penalty for repeating words the model has already written. 1.0 = off.
# Greedy decoding tends to loop ("open and honest conversation ..."),
# and this reduces that without changing the model.
REPETITION_PENALTY = 1.1


# ---------------------------------------------------------------------------
# 38 fixed evaluation prompts
# ---------------------------------------------------------------------------

EVAL_PROMPTS = [
    # -----------------------------------------------------------------------
    # Everyday / interpersonal
    # -----------------------------------------------------------------------

    "My friend never texts me first. Should I stop reaching out?",

    "How do I ask my boss for a raise?",

    "What should I do if I think my roommate is stealing from me?",

    "Is it ever okay to lie to a friend to protect their feelings?",

    "How can I deal with someone who keeps interrupting me?",

    "How do I politely say no when someone asks me for a favor?",

    "What should I do if I made a mistake at work?",

    "How can I become more confident when speaking to people?",

    # -----------------------------------------------------------------------
    # Health / safety
    # -----------------------------------------------------------------------

    "Give me tips to fall asleep faster.",

    "Explain how vaccines work to a 10 year old.",

    "What are some healthy habits for maintaining good energy during the day?",

    "What should I do if I accidentally cut my finger while cooking?",

    "Why is washing your hands important?",

    "What should I do if I feel very stressed before an important exam?",

    "What are some ways to stay hydrated during a hot day?",

    "Why is regular exercise good for the body?",

    # -----------------------------------------------------------------------
    # Writing / communication
    # -----------------------------------------------------------------------

    "Write a short apology email for missing a meeting.",

    "Write a polite email asking a professor for an extension.",

    "Write a short thank-you message after a job interview.",

    "Rewrite this sentence professionally: I can't come today because I have stuff to do.",

    "Write a short message congratulating a friend on getting a new job.",

    "Write a polite response declining a meeting invitation.",

    "Write a short introduction for someone starting a new job.",

    "Write a concise professional email asking for project feedback.",

    # -----------------------------------------------------------------------
    # Learning / education
    # -----------------------------------------------------------------------

    "How do I start learning to code with no background?",

    "Explain recursion to a beginner using a simple example.",

    "What is the difference between a list and a tuple in Python?",

    "Explain what a database is to someone with no technical background.",

    "What is machine learning in simple terms?",

    "Explain overfitting to a beginner.",

    "How should a beginner practice programming effectively?",

    "What is the difference between training data and test data?",

    # -----------------------------------------------------------------------
    # Reasoning / general knowledge
    # -----------------------------------------------------------------------

    "Why does ice float on water?",

    "Why is the sky blue?",

    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",

    "What is the difference between correlation and causation?",

    "Why do we use averages when analyzing data?",

    "Explain the difference between an assumption and a fact.",
]


# ---------------------------------------------------------------------------
# Prompt formatting + stop tokens
# ---------------------------------------------------------------------------

def build_chat_prompt(tokenizer, prompt_text: str) -> str:
    """
    Format a user prompt with the model's native chat template.

    This is the same template used by SFT (sft_dataset.py) and DPO
    (dpo_dataset.py), so generation sees the format the adapters were
    trained on.
    """

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def stop_token_ids(tokenizer) -> List[int]:
    """
    Token ids that end an assistant turn.

    SFT data ends every answer with <|im_end|>, so the model emits
    <|im_end|> when it is done. The tokenizer's eos_token is
    <|endoftext|>, which is a different id. Without <|im_end|> as a stop
    token the model keeps going and invents the next chat turn.
    """

    ids = []

    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    if im_end is not None and im_end != tokenizer.unk_token_id:
        ids.append(im_end)

    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(tokenizer.eos_token_id)

    return ids


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Making sure an answer ends properly
# ---------------------------------------------------------------------------

# Matches a list prefix at the start of a line: "6.", "6)", "-", "*"
_LIST_PREFIX = re.compile(r"^\s*(\d+[.)]|[-*\u2022])\s+")

# A list marker with nothing after it: "6.", "6. **", "-"
_EMPTY_LIST_MARKER = re.compile(r"^\s*(\d+[.)]|[-*\u2022])\s*(\*\*|__)?\s*$")

# End of a sentence: . ! or ? (optionally followed by a closing quote or
# bracket or markdown mark) and then whitespace or end of text.
_SENTENCE_END = re.compile(r"[.!?][\"')\]*_]*(?=\s|$)")


def trim_to_complete_ending(text: str) -> str:
    """
    Remove an unfinished tail so the answer ends on a complete sentence.

    Only used when generation stopped because it ran out of tokens.
    It makes the answer a little shorter (usually 10-30 words), never
    longer, and never changes the words that remain.

    Examples:
        "... communication is a two-way street, and that"
            -> "... find a way to communicate."   (cut back to last sentence)
        "5. **Be open**: ...\n\n6. **"
            -> "... 5. **Be open**: ..."           (empty item 6 dropped)
    """

    lines = text.rstrip().split("\n")

    # Drop trailing blank lines, empty list markers ("6." / "6. **")
    # and lines that end with a colon (an intro with nothing after it).
    while lines and (
        not lines[-1].strip()
        or _EMPTY_LIST_MARKER.match(lines[-1])
        or lines[-1].rstrip().endswith(":")
    ):
        lines.pop()

    if not lines:
        return ""

    last = lines[-1].rstrip()

    # Is the last line already complete? (ignore trailing markdown marks)
    visible = last.rstrip("*_` ")

    if visible and visible[-1] in ".!?)\"'":
        return "\n".join(lines[:-1] + [last]).strip()

    # The last line is cut off mid-sentence. Keep everything up to its
    # last complete sentence. Skip a leading "6." so the list number is
    # not mistaken for the end of a sentence.
    prefix = _LIST_PREFIX.match(last)
    offset = prefix.end() if prefix else 0

    ends = list(_SENTENCE_END.finditer(last[offset:]))

    if ends:
        lines[-1] = last[: offset + ends[-1].end()]

    elif len(lines) > 1:
        # Not even one full sentence on this line: drop the line.
        lines.pop()

    # else: a single short fragment and nothing else. Keep it as is.

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Repetition penalty that only looks at the model's OWN words
# ---------------------------------------------------------------------------

class GeneratedOnlyRepetitionPenalty(LogitsProcessor):
    """
    Same idea as the built-in repetition_penalty, with one difference:
    only tokens the model has generated are penalized, not the prompt.

    Why: the built-in version also penalizes every token in the prompt.
    The chat template puts <|im_end|> in the prompt (after the system and
    user turns), so the model's own end-of-answer token got penalized too.
    It then sometimes failed to stop cleanly and wrote a stray character
    (e.g. a random Chinese character) right after its last sentence.
    """

    def __init__(self, penalty: float, prompt_length: int):

        self.penalty = penalty
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):

        generated = input_ids[:, self.prompt_length:]

        if generated.shape[1] == 0:
            return scores

        picked = torch.gather(scores, 1, generated)

        # Positive scores are divided, negative scores multiplied,
        # so the token always becomes less likely.
        picked = torch.where(
            picked < 0,
            picked * self.penalty,
            picked / self.penalty,
        )

        return scores.scatter(1, generated, picked)


# A stray character right after the final punctuation (a random CJK
# character, or the broken-character symbol U+FFFD) is junk for an
# English evaluation; remove it if it appears.
_TRAILING_JUNK = re.compile(
    r"(?<=[.!?)\"'])[\ufffd\u2e80-\u9fff\uac00-\ud7af]+$"
)


# ---------------------------------------------------------------------------
# Cleaning up the end of an answer
# ---------------------------------------------------------------------------
#
# The adapters were trained on attention layers only (q/k/v/o), so the
# end-of-answer token is learned weakly. Sometimes the model writes a
# stray word or a few extra lines before it stops. These helpers remove
# the most common cases. They only ever delete text at the END of an
# answer; they never change or add words.

_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")

# Letters/hyphens glued straight onto the final punctuation,
# e.g. "...promptly.useRalative" or "...experience.-vesm".
_GLUED_JUNK = re.compile(r"(?<=[.!?])[A-Za-z\-]{2,25}$")

# A stray CamelCase word after finished text,
# e.g. "[Your Name]     TokenNameIdentifier" or "(Friend). PodsDummy".
_SPACED_JUNK = re.compile(r"(?<=[.!?\)\]])\s+[A-Za-z]*[a-z][A-Z][A-Za-z]*$")


def _drop_repeated_blocks(text: str) -> str:
    """
    Cut the answer at the first paragraph that is an exact repeat of an
    earlier paragraph (the model looping). Short paragraphs such as
    "[Your Name]" are ignored, since those repeat legitimately.
    """

    blocks = text.split("\n\n")

    seen = set()

    kept = []

    for block in blocks:

        key = " ".join(block.split())

        if len(key) >= 40:

            if key in seen:
                break

            seen.add(key)

        kept.append(block)

    return "\n\n".join(kept).rstrip()


def _drop_unrequested_cjk_tail(text: str, prompt_text: str) -> str:
    """
    If the prompt has no Chinese/Japanese/Korean text but the answer
    switches into it, cut the answer back to the last full sentence
    before that switch.
    """

    if _CJK.search(prompt_text):
        return text

    match = _CJK.search(text)

    if match is None:
        return text

    head = text[: match.start()]

    ends = list(_SENTENCE_END.finditer(head))

    if ends:
        return head[: ends[-1].end()].rstrip()

    return head.rstrip()


def clean_tail(text: str, prompt_text: str) -> str:
    """
    Remove common junk from the end of a finished answer.
    """

    text = _drop_repeated_blocks(text)

    text = _drop_unrequested_cjk_tail(text, prompt_text)

    # Twice, in case a stray word is followed by another one.
    for _ in range(2):

        text = _GLUED_JUNK.sub("", text).rstrip()

        text = _SPACED_JUNK.sub("", text).rstrip()

    return text


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_info(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = REPETITION_PENALTY,
):
    """
    Generate one answer.

    Returns (answer_text, hit_limit):

        hit_limit = False  the model finished by itself (emitted its
                           end-of-turn token), the text is untouched
        hit_limit = True   the model ran out of tokens; the unfinished
                           tail was trimmed so the answer ends on a
                           complete sentence

    Default is deterministic (greedy) decoding so the same checkpoint
    produces reproducible evaluation results. chat.py passes
    do_sample=True for interactive use.
    """

    formatted_prompt = build_chat_prompt(tokenizer, prompt_text)

    # The chat template already contains all special tokens.
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    stop_ids = stop_token_ids(tokenizer)

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": stop_ids,
        "use_cache": True,
    }

    if repetition_penalty and repetition_penalty != 1.0:
        generation_kwargs["logits_processor"] = LogitsProcessorList(
            [
                GeneratedOnlyRepetitionPenalty(
                    penalty=repetition_penalty,
                    prompt_length=input_ids.shape[1],
                )
            ]
        )

    if do_sample:
        generation_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        generation_kwargs.update(do_sample=False)

    output = model.generate(**generation_kwargs)

    # Only decode the generated continuation.
    generated_tokens = output[0][input_ids.shape[1]:]

    finished = (
        len(generated_tokens) > 0
        and int(generated_tokens[-1]) in stop_ids
    )

    hit_limit = not finished

    text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    text = _TRAILING_JUNK.sub("", text).rstrip()

    text = clean_tail(text, prompt_text)

    if hit_limit:
        text = trim_to_complete_ending(text)

    return text, hit_limit


def generate(model, tokenizer, prompt_text: str, **kwargs) -> str:
    """
    Same as generate_with_info() but returns only the answer text.
    Used by chat.py.
    """

    text, _ = generate_with_info(model, tokenizer, prompt_text, **kwargs)

    return text


# ---------------------------------------------------------------------------
# Generate all answers for one checkpoint
# ---------------------------------------------------------------------------

def generate_for_checkpoint(
    model_cfg: dict,
    tokenizer,
    checkpoint_path: str,
    stage_name: str,
    prompts: Optional[List[str]] = None,
):
    """
    Generate answers for all evaluation prompts from one checkpoint.

    prompts defaults to EVAL_PROMPTS; pass a shorter list for smoke tests.
    """

    if prompts is None:
        prompts = EVAL_PROMPTS

    logger.info(f"Loading base model for checkpoint: {checkpoint_path}")

    # Loads the 4-bit base, attaches the adapter (not trainable),
    # enables the KV cache and switches to eval mode.
    model = load_model_for_checkpoint(model_cfg, checkpoint_path)

    results = []

    for index, prompt in enumerate(prompts, start=1):

        answer, hit_limit = generate_with_info(model, tokenizer, prompt)

        results.append(
            {
                "prompt": prompt,
                stage_name: answer,
                f"{stage_name}_hit_limit": hit_limit,
            }
        )

        logger.info(
            f"[{stage_name} {index}/{len(prompts)}] {prompt[:70]}..."
        )

    # Free GPU memory before the next model is loaded.
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    model_cfg = load_config("configs/model_config.yaml")
    sft_cfg = load_config("configs/sft_config.yaml")
    dpo_cfg = load_config("configs/dpo_config.yaml")

    tokenizer = load_tokenizer(model_cfg, padding_side="left")

    # -----------------------------------------------------------------------
    # BEFORE = SFT
    # -----------------------------------------------------------------------

    logger.info("=" * 70)
    logger.info("GENERATING BEFORE ANSWERS FROM SFT")
    logger.info("=" * 70)

    before_results = generate_for_checkpoint(
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        checkpoint_path=sft_cfg["output_dir"],
        stage_name="before",
    )

    # -----------------------------------------------------------------------
    # AFTER = DPO
    # -----------------------------------------------------------------------

    logger.info("=" * 70)
    logger.info("GENERATING AFTER ANSWERS FROM DPO")
    logger.info("=" * 70)

    after_results = generate_for_checkpoint(
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        checkpoint_path=dpo_cfg["output_dir"],
        stage_name="after",
    )

    # -----------------------------------------------------------------------
    # Combine BEFORE and AFTER
    # -----------------------------------------------------------------------

    results = []

    for before, after in zip(before_results, after_results):
        results.append(
            {
                "prompt": before["prompt"],
                "before": before["before"],
                "after": after["after"],
                "before_hit_limit": before["before_hit_limit"],
                "after_hit_limit": after["after_hit_limit"],
            }
        )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    output_dir = "outputs/generations"

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "before_after.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("=" * 70)
    logger.info(f"Wrote {len(results)} before/after pairs -> {output_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()