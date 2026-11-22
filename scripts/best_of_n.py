"""How much could picking the best of the model's sampled replies raise the score? No model or API calls.

    python scripts/best_of_n.py outputs/stage2/candidates.jsonl outputs/stage4/candidates.jsonl \
        outputs/stage4_photo/candidates.jsonl

Each candidates.jsonl (written by `scrable.dpo pairs`) holds, per training review, the 1.5B's sampled
replies ("policy", temperature 0.8) and Haiku's reply, all scored by the Haiku judge. Per review this
compares ways of picking one of the 1.5B replies:

  random       the average of its replies (what serving one sample gives)
  best         the highest-scoring reply: the ceiling of any picker (needs the judge, so not usable live)
  longest      the longest reply
  doc words    the reply using the most words from the retrieved store documents that the review
               itself does not contain (a crude, free stand-in for "mentions store specifics")
  haiku        Haiku's reply, for reference

Gains are against random, with a 95% bootstrap interval over reviews. Caveats: the sampled replies
come from the base 1.5B with the prompt (no adapter), not from the trained server model, which
usually varies less between samples; and these are training reviews.
"""

import argparse
import json
import random
import re
from pathlib import Path

WORD = re.compile(r"[0-9A-Za-z가-힣]{2,}")


def specificity(candidate):
    return (candidate["scores"]["specificity"] - 1) / 4


def user_payload(messages):
    content = next((m["content"] for m in messages if m["role"] == "user"), "")
    if isinstance(content, list):  # image part + text part
        content = next((p.get("text", "") for p in content if p.get("type") == "text"), "")
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return {}


def doc_words(row):
    payload = user_payload(row.get("messages") or [])
    docs = " ".join(d.get("text", "") for d in payload.get("retrieved_documents", []))
    return set(WORD.findall(docs)) - set(WORD.findall(row.get("review") or ""))


def picks(row):
    """{picker: candidate} for one review, or None when it has fewer than 2 distinct 1.5B replies."""
    policy = [c for c in row["candidates"] if c["source"] == "policy"]
    if len(policy) < 2:
        return None
    words = doc_words(row)
    teacher = next((c for c in row["candidates"] if c["source"].startswith("teacher")), None)
    return {
        "policy": policy,
        "best": max(policy, key=lambda c: c["score"]),
        "longest": max(policy, key=lambda c: len(c["text"])),
        "doc words": max(policy, key=lambda c: len(words & set(WORD.findall(c["text"])))),
        "haiku": teacher,
    }


def interval(values, seed=0, rounds=2000):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(rounds))
    return means[int(rounds * 0.025)], means[int(rounds * 0.975)]


def summarize(rows):
    chosen = [p for p in map(picks, rows) if p is not None]
    if not chosen:
        raise ValueError("No review has at least 2 distinct 1.5B replies")
    base = [sum(c["score"] for c in p["policy"]) / len(p["policy"]) for p in chosen]
    base_spec = [sum(specificity(c) for c in p["policy"]) / len(p["policy"]) for p in chosen]
    out = {
        "reviews": len(chosen),
        "skipped": len(rows) - len(chosen),
        "random": (sum(base) / len(base), sum(base_spec) / len(base_spec), None),
    }
    for name in ("best", "longest", "doc words", "haiku"):
        subset = [(p[name], b, bs) for p, b, bs in zip(chosen, base, base_spec) if p[name] is not None]
        if not subset:
            continue
        gains = [c["score"] - b for c, b, _ in subset]
        out[name] = (
            sum(c["score"] for c, _, _ in subset) / len(subset),
            sum(specificity(c) for c, _, _ in subset) / len(subset),
            (sum(gains) / len(gains), *interval(gains)),
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidates", nargs="+", help="candidates.jsonl files from scrable.dpo pairs")
    args = parser.parse_args()
    for path in args.candidates:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        result = summarize(rows)
        print(f"\n== {path}: reviews {result['reviews']} (skipped {result['skipped']}: fewer than 2 distinct replies)")
        print(f"  {'picker':10} overall  specificity  gain vs random [95%]")
        for name in ("random", "best", "longest", "doc words", "haiku"):
            if name not in result:
                continue
            overall, spec, gain = result[name]
            text = "" if gain is None else f"{gain[0]:+.3f} [{gain[1]:+.3f}, {gain[2]:+.3f}]"
            print(f"  {name:10} {overall:.3f}    {spec:.3f}        {text}")


if __name__ == "__main__":
    main()
