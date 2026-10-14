"""
SFT vs DPO comparison.

Default mode shows BOTH answers side by side for every prompt (no judge,
no win/loss/tie). You read the two answers and decide.

    python -m src.eval.compare                # side-by-side only
    python -m src.eval.compare --limit 3      # quick smoke test
    python -m src.eval.compare --stride 2     # every 2nd prompt (19 of 38)
    python -m src.eval.compare --judge        # also run the local LLM judge

Outputs:
    outputs/eval_reports/side_by_side.md           readable report (open this)
    outputs/eval_reports/before_after_report.jsonl full data
"""

import argparse
import gc
import json
import os
import re
from statistics import mean

import torch

from src.eval.generate_answers import EVAL_PROMPTS, generate_for_checkpoint
from src.models.load_model import load_tokenizer
from src.utils.config import load_config
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


# The judge works with positions: A = SFT/before, B = DPO/after.
WINNER_LABELS = {
    "A": "SFT",
    "B": "DPO",
    "TIE": "TIE",
    "UNCERTAIN": "UNCERTAIN",
}

REPORT_DIR = "outputs/eval_reports"
TERMINAL_PREVIEW_CHARS = 4000


# ---------------------------------------------------------------------------
# Simple, judge-free answer statistics
# ---------------------------------------------------------------------------

def answer_stats(text: str, hit_limit: bool = False) -> dict:
    """
    Cheap objective facts about an answer.

    words         length of the answer
    ends_cleanly  True if the last line is a finished sentence. A bare
                  list marker such as "6." or "6. **" counts as NOT clean.
    unique_ratio  unique words / total words. Low values mean repetition.
    hit_limit     True if the model ran out of tokens. The unfinished
                  tail was then trimmed, so the text is shorter than
                  what the model wanted to write.
    """

    words = text.split()

    if not words:
        return {
            "words": 0,
            "ends_cleanly": False,
            "unique_ratio": 0.0,
            "hit_limit": hit_limit,
        }

    last_line = text.rstrip().split("\n")[-1]

    # Ignore trailing markdown marks when looking at the final character.
    visible = last_line.rstrip("*_` ")

    empty_marker = re.match(r"^\s*(\d+[.)]|[-*\u2022])\s*$", visible) is not None

    return {
        "words": len(words),
        "ends_cleanly": (
            bool(visible) and visible[-1] in '.!?)"\'' and not empty_marker
        ),
        "unique_ratio": round(len(set(w.lower() for w in words)) / len(words), 2),
        "hit_limit": hit_limit,
    }


def _stats_line(label: str, stats: dict) -> str:

    return (
        f"{label}: {stats['words']} words | "
        f"ends cleanly: {'yes' if stats['ends_cleanly'] else 'NO'} | "
        f"unique-word ratio: {stats['unique_ratio']} | "
        f"hit token limit (trimmed): {'yes' if stats['hit_limit'] else 'no'}"
    )


def _preview(text: str) -> str:

    if len(text) <= TERMINAL_PREVIEW_CHARS:
        return text

    return text[:TERMINAL_PREVIEW_CHARS].rstrip() + " ..."


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def build_markdown(records: list, judged: bool) -> str:

    lines = ["# SFT vs DPO: side-by-side answers", ""]

    for i, record in enumerate(records, start=1):

        lines.append(f"## {i}. {record['prompt']}")
        lines.append("")

        lines.append("### SFT (before)")
        lines.append("")
        lines.append(record["before"] or "_(empty answer)_")
        lines.append("")
        lines.append(f"_{_stats_line('SFT', record['stats_before'])}_")
        lines.append("")

        lines.append("### DPO (after)")
        lines.append("")
        lines.append(record["after"] or "_(empty answer)_")
        lines.append("")
        lines.append(f"_{_stats_line('DPO', record['stats_after'])}_")
        lines.append("")

        if judged:

            judge = record["judge"]

            if judge.get("valid"):
                lines.append(
                    f"_Judge scores (overall): SFT {judge['a']['overall']} | "
                    f"DPO {judge['b']['overall']} | "
                    f"verdict: {record['winner']}_"
                )
            else:
                lines.append("_Judge: no valid score for this pair_")

            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional judge stage
# ---------------------------------------------------------------------------

def run_judge(records: list):

    # Imported only when --judge is used, so the default run does not
    # load a third model.
    from src.eval.judge import judge_pair, load_judge

    logger.info("=" * 70)
    logger.info("OPTIONAL STAGE: independent local LLM judging")
    logger.info("=" * 70)

    judge_model, judge_tokenizer = load_judge()

    for index, record in enumerate(records, start=1):

        logger.info(f"Judging pair {index}/{len(records)}...")

        score = judge_pair(
            model=judge_model,
            tokenizer=judge_tokenizer,
            question=record["prompt"],
            answer_before=record["before"],
            answer_after=record["after"],
        )

        record["judge"] = score
        record["winner"] = WINNER_LABELS.get(score["winner"], "UNCERTAIN")

    del judge_model
    del judge_tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="SFT vs DPO comparison")

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N prompts (smoke test).",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth prompt (e.g. 2 = 19 of the 38 prompts, "
             "still covering every category). Faster than the full set.",
    )

    parser.add_argument(
        "--judge",
        action="store_true",
        help="Also run the local LLM judge (off by default).",
    )

    args = parser.parse_args()

    prompts = EVAL_PROMPTS

    if args.stride > 1:
        prompts = prompts[:: args.stride]

    if args.limit is not None:
        prompts = prompts[: args.limit]

    model_cfg = load_config("configs/model_config.yaml")
    sft_cfg = load_config("configs/sft_config.yaml")
    dpo_cfg = load_config("configs/dpo_config.yaml")

    tokenizer = load_tokenizer(model_cfg, padding_side="left")

    # -----------------------------------------------------------------------
    # Generate BEFORE (SFT) and AFTER (DPO) answers
    # -----------------------------------------------------------------------

    logger.info("=" * 70)
    logger.info("STAGE 1: Generating BEFORE answers from SFT checkpoint")
    logger.info("=" * 70)

    before_results = generate_for_checkpoint(
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        checkpoint_path=sft_cfg["output_dir"],
        stage_name="before",
        prompts=prompts,
    )

    logger.info("=" * 70)
    logger.info("STAGE 2: Generating AFTER answers from DPO checkpoint")
    logger.info("=" * 70)

    after_results = generate_for_checkpoint(
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        checkpoint_path=dpo_cfg["output_dir"],
        stage_name="after",
        prompts=prompts,
    )

    records = []

    for before, after in zip(before_results, after_results):

        records.append(
            {
                "prompt": before["prompt"],
                "before": before["before"],
                "after": after["after"],
                "before_hit_limit": before["before_hit_limit"],
                "after_hit_limit": after["after_hit_limit"],
                "stats_before": answer_stats(
                    before["before"], before["before_hit_limit"]
                ),
                "stats_after": answer_stats(
                    after["after"], after["after_hit_limit"]
                ),
            }
        )

    # -----------------------------------------------------------------------
    # Optional judge
    # -----------------------------------------------------------------------

    if args.judge:
        run_judge(records)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    os.makedirs(REPORT_DIR, exist_ok=True)

    jsonl_path = os.path.join(REPORT_DIR, "before_after_report.jsonl")
    md_path = os.path.join(REPORT_DIR, "side_by_side.md")

    with open(jsonl_path, "w", encoding="utf-8") as f:

        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown(records, judged=args.judge))

    # -----------------------------------------------------------------------
    # Terminal: both answers for every prompt
    # -----------------------------------------------------------------------

    print()

    for i, record in enumerate(records, start=1):

        print("=" * 90)
        print(f"{i}. {record['prompt']}")
        print("=" * 90)

        print()
        print("SFT (before):")
        print(_preview(record["before"]))
        print(_stats_line("SFT", record["stats_before"]))

        print()
        print("DPO (after):")
        print(_preview(record["after"]))
        print(_stats_line("DPO", record["stats_after"]))

        if args.judge and record["judge"].get("valid"):
            print()
            print(
                f"Judge overall: SFT {record['judge']['a']['overall']} | "
                f"DPO {record['judge']['b']['overall']} | {record['winner']}"
            )

        print()

    # -----------------------------------------------------------------------
    # Summary of the objective stats
    # -----------------------------------------------------------------------

    print("=" * 90)
    print("SUMMARY (judge-free)")
    print("=" * 90)

    for label, key in (("SFT", "stats_before"), ("DPO", "stats_after")):

        stats = [r[key] for r in records]

        not_clean = sum(1 for s in stats if not s["ends_cleanly"])
        hit_limit = sum(1 for s in stats if s["hit_limit"])

        print(
            f"{label}: avg {mean(s['words'] for s in stats):.0f} words | "
            f"avg unique-word ratio {mean(s['unique_ratio'] for s in stats):.2f} | "
            f"hit token limit: {hit_limit}/{len(stats)} | "
            f"not ending cleanly: {not_clean}/{len(stats)}"
        )

    if args.judge:

        valid = [r for r in records if r["judge"].get("valid")]

        print()
        print(f"Judge gave a valid score for {len(valid)}/{len(records)} pairs.")

        if valid:
            print(
                f"Mean judge overall: "
                f"SFT {mean(r['judge']['a']['overall'] for r in valid):.2f} | "
                f"DPO {mean(r['judge']['b']['overall'] for r in valid):.2f}"
            )

    print()
    print(f"Readable report: {md_path}")
    print(f"Full data:       {jsonl_path}")
    print()


if __name__ == "__main__":
    main()