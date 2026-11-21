"""How often does each photo-description setup name the food right? No model or API calls.

    python scripts/photo_naming.py --reviews data/photo_reviews_v2.jsonl \
        Vision-3B=outputs/photo_eval_stage4/all_predictions.json \
        Vision-3B+메뉴=outputs/photo_eval_context2/all_predictions.json \
        Qwen=outputs/photo_qwen/all_predictions.json

Each NAME=FILE is an earlier evaluation's predictions (they keep the photo description of every
review). Descriptions are matched to photos through the review file, and checked against
configs/photo_labels.json: right when the description contains an accepted word. Photos marked
use=false are listed but not counted. The answer is only as good as the labels; read the
printed descriptions for the misses.
"""

import argparse
import json
from pathlib import Path


def descriptions(predictions_path, image_of):
    """{photo file name: first description seen}."""
    out = {}
    for p in json.loads(Path(predictions_path).read_text(encoding="utf-8"))["predictions"]:
        image = image_of.get(p["review_id"])
        if image and p.get("photo_description") and image not in out:
            out[image] = p["photo_description"]
    return out


def score(descs, labels):
    rows, right, counted = [], 0, 0
    for name, label in labels.items():
        text = descs.get(name, "")
        ok = any(word in text for word in label["accept"]) if text else False
        if label["use"] and text:
            counted += 1
            right += ok
        rows.append((name, label["use"], ok, text))
    return right, counted, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="NAME=predictions.json")
    parser.add_argument("--reviews", default="data/photo_reviews_v2.jsonl")
    parser.add_argument("--labels", default="configs/photo_labels.json")
    parser.add_argument("--show", type=int, default=70, help="Characters of each description to print")
    args = parser.parse_args()
    labels = {
        k: v
        for k, v in json.loads(Path(args.labels).read_text(encoding="utf-8")).items()
        if not k.startswith("_")
    }
    image_of = {}
    for line in Path(args.reviews).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("image"):
                image_of[row["id"]] = Path(row["image"]).name
    summary = []
    for run in args.runs:
        name, _, path = run.partition("=")
        right, counted, rows = score(descriptions(path, image_of), labels)
        summary.append((name, right, counted))
        print(f"\n== {name}: {right}/{counted} 맞음")
        for photo, use, ok, text in rows:
            mark = "-" if not use else ("O" if ok else "X")
            print(f"  {mark} {photo:13} {labels[photo]['truth'][:18]:18} | {text[: args.show]}")
    print("\n요약")
    for name, right, counted in summary:
        print(f"  {name}: {right}/{counted}")


if __name__ == "__main__":
    main()
