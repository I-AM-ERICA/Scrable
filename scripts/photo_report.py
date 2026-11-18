"""Show each photo review with the photo description and both replies, side by side.

    python scripts/photo_report.py outputs/photo_eval/with_photos/all_predictions.json \
        outputs/photo_eval/no_photos/all_predictions.json

Starts with a table per review style (photo / short / complaint, see scrable/synth_photo_reviews.py):
mean score and "acc<=2", the replies the judge gave accuracy 2 or less (usually a wrong or
invented claim). Add --summary to print only the table. Scores are LLM-judge scores (the judge saw
the photo), not human evaluation.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scrable.synth_photo_reviews import style_of  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("label", Path(path).parent.name), {r["review_id"]: r for r in data["predictions"]}


def summary(runs):
    styles = sorted({style_of(i) for i in runs[0][1]})
    print("style".ljust(12) + "".join(f"{label:>22}" for label, _ in runs))
    print(" " * 12 + "".join(f"{'score / acc<=2':>22}" for _ in runs))
    for style in [*styles, "all"]:
        cells = []
        for _, run in runs:
            rows = [r for i, r in run.items() if style == "all" or style_of(i) == style]
            mean = sum(r["score"] for r in rows) / len(rows)
            low = sum(r["scores"]["accuracy"] <= 2 for r in rows)
            cells.append(f"{mean:.3f} / {low}/{len(rows)}")
        print(f"{style:<12}" + "".join(f"{c:>22}" for c in cells))


def main(paths, only_summary=False):
    runs = [load(p) for p in paths]
    summary(runs)
    if only_summary:
        return
    first = runs[0][1]
    for review_id in sorted(first):
        row = first[review_id]
        print(f"\n===== {review_id}\n[리뷰] {row['review']}")
        description = next(
            (r[review_id].get("photo_description") for _, r in runs if r[review_id].get("photo_description")),
            None,
        )
        print(f"[사진 설명] {description or '(없음)'}")
        for label, run in runs:
            other = run[review_id]
            scores = " ".join(f"{k[:3]} {v}" for k, v in other["scores"].items())
            print(f"[{label}] ({other['score']:.2f}; {scores})\n{other['prediction']}")


if __name__ == "__main__":
    files = [a for a in sys.argv[1:] if a != "--summary"]
    if not files:
        raise SystemExit(__doc__)
    main(files, "--summary" in sys.argv)
