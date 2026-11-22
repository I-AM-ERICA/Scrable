"""Photo-facts experiment: give the reply model person-checked photo facts instead of Vision-3B's
description, and check reply by reply whether the photo's information was used. No model or API
calls. Steps are in RUN_PHOTO_FACTS.md.

    python scripts/photo_facts.py build
        configs/photo_facts.json (+ data/photo_reviews_v2.jsonl) -> outputs/photo_facts/descriptions.json,
        a predictions-shaped file that configs/photo_eval_facts.json reads as fixed photo descriptions
        (photo_descriptions_from). Refuses facts a person has not checked.

    python scripts/photo_facts.py sheet --vision outputs/photo_eval_stage4/all_predictions.json \
        --facts-run outputs/photo_eval_facts/all_predictions.json \
        [--none outputs/photo_eval_nophoto/all_predictions.json] > facts_sheet.csv
        One row per review: review, checked facts, Vision-3B description, the replies, and empty
        예/아니오 columns for a person to fill in (UTF-8 with BOM, opens in Excel).

    python scripts/photo_facts.py tally facts_sheet.csv
        Counts the filled columns: among reviews where the photo adds useful information the
        review text does not have, how often the Vision-3B description has it and how often each
        reply uses it correctly; and unsupported claims in each reply.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ADDS = "사진이 리뷰에 없는 쓸 만한 정보를 줌"
VISION_HAS = "② 설명에 그 정보가 있음"
VISION_USED = "② 답글이 그 정보를 맞게 씀"
FACTS_USED = "③ 답글이 그 정보를 맞게 씀"
VISION_UNSUPPORTED = "② 답글에 근거 없는 주장"
FACTS_UNSUPPORTED = "③ 답글에 근거 없는 주장"
FILL_IN = [ADDS, VISION_HAS, VISION_USED, FACTS_USED, VISION_UNSUPPORTED, FACTS_UNSUPPORTED, "메모"]
YES = {"예", "네", "y", "yes", "o", "1", "true"}
NO = {"아니오", "아니요", "n", "no", "x", "0", "false"}


def load_facts(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def photo_of_reviews(reviews_path):
    """{review id: photo file name} for the reviews that have a photo."""
    out = {}
    for line in Path(reviews_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("image"):
                out[row["id"]] = Path(row["image"]).name
    return out


def build(facts, photo_of, allow_unchecked=False):
    """Predictions-shaped dict ({"predictions": [{review_id, photo_description}]})."""
    unchecked = sorted(name for name, entry in facts.items() if not entry.get("checked"))
    if unchecked and not allow_unchecked:
        raise ValueError(
            f"{len(unchecked)} photos are not checked yet ({', '.join(unchecked[:4])}...): "
            "look at each photo, fix the facts, set checked to true"
        )
    missing = sorted({name for name in photo_of.values() if name not in facts})
    if missing:
        raise ValueError(f"No facts for photos: {', '.join(missing)}")
    empty = sorted(name for name, entry in facts.items() if not str(entry.get("facts", "")).strip())
    if empty:
        raise ValueError(f"Empty facts for photos: {', '.join(empty)}")
    return {
        "_comment": "Person-checked photo facts (configs/photo_facts.json) per review, read by "
        "configs/photo_eval_facts.json as fixed photo descriptions.",
        "predictions": [
            {"review_id": review_id, "photo_description": facts[name]["facts"].strip()}
            for review_id, name in sorted(photo_of.items())
        ],
    }


def by_review(path):
    if not path:
        return {}
    return {p["review_id"]: p for p in json.loads(Path(path).read_text(encoding="utf-8"))["predictions"]}


def sheet_rows(vision, facts_run, none=None):
    """Rows in the order of the Vision-3B run; only reviews present in both runs."""
    columns = ["review_id", "리뷰", "사진", "③ 사람 확인 사실", "② Vision-3B 설명"]
    if none:
        columns += ["① 답글 (사진 정보 없음)", "① 점수"]
    columns += ["② 답글", "② 점수", "③ 답글", "③ 점수"] + FILL_IN
    rows = []
    for review_id, v in vision.items():
        f = facts_run.get(review_id)
        if f is None:
            continue
        row = {
            "review_id": review_id,
            "리뷰": v["review"],
            "사진": Path(v.get("image") or f.get("image") or "").name,
            "③ 사람 확인 사실": f.get("photo_description") or "",
            "② Vision-3B 설명": v.get("photo_description") or "",
        }
        if none:
            n = none.get(review_id, {})
            row["① 답글 (사진 정보 없음)"] = n.get("prediction", "")
            row["① 점수"] = round(n["score"], 3) if "score" in n else ""
        row |= {
            "② 답글": v["prediction"],
            "② 점수": round(v["score"], 3),
            "③ 답글": f["prediction"],
            "③ 점수": round(f["score"], 3),
        }
        row |= {name: "" for name in FILL_IN}
        rows.append(row)
    return columns, rows


def answer(value):
    text = str(value or "").strip().lower()
    if text in YES:
        return True
    if text in NO:
        return False
    return None


def read_sheet(path):
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp949"):  # Excel's plain "CSV" save on Korean Windows is cp949.
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Save the sheet as CSV (UTF-8 or the default Excel CSV)")
    return list(csv.DictReader(text.splitlines()))


def tally(rows):
    """Counts over filled rows; blanks are left out and reported."""

    def rate(subset, column):
        values = [answer(r.get(column)) for r in subset]
        known = [v for v in values if v is not None]
        return sum(known), len(known), len(values) - len(known)

    useful = [r for r in rows if answer(r.get(ADDS)) is True]
    return {
        "reviews": len(rows),
        "photo_adds_info": rate(rows, ADDS),
        "vision_has_it": rate(useful, VISION_HAS),
        "vision_reply_uses_it": rate(useful, VISION_USED),
        "facts_reply_uses_it": rate(useful, FACTS_USED),
        "vision_reply_unsupported": rate(rows, VISION_UNSUPPORTED),
        "facts_reply_unsupported": rate(rows, FACTS_UNSUPPORTED),
    }


def print_tally(result):
    labels = {
        "photo_adds_info": "사진이 리뷰에 없는 쓸 만한 정보를 줌 (전체 중)",
        "vision_has_it": "② Vision-3B 설명에 그 정보가 있음 (정보를 준 리뷰 중)",
        "vision_reply_uses_it": "② 답글이 그 정보를 맞게 씀 (정보를 준 리뷰 중)",
        "facts_reply_uses_it": "③ 답글이 그 정보를 맞게 씀 (정보를 준 리뷰 중)",
        "vision_reply_unsupported": "② 답글에 근거 없는 주장 (전체 중)",
        "facts_reply_unsupported": "③ 답글에 근거 없는 주장 (전체 중)",
    }
    print(f"리뷰 {result['reviews']}건")
    for key, label in labels.items():
        yes, known, blank = result[key]
        note = f" (빈칸 {blank})" if blank else ""
        print(f"  {label}: {yes}/{known}{note}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--facts", default="configs/photo_facts.json")
    b.add_argument("--reviews", default="data/photo_reviews_v2.jsonl")
    b.add_argument("--out", default="outputs/photo_facts/descriptions.json")
    b.add_argument("--allow-unchecked", action="store_true", help="Pilot only: use unchecked drafts")
    s = sub.add_parser("sheet")
    s.add_argument("--vision", required=True, help="② predictions (Vision-3B descriptions)")
    s.add_argument("--facts-run", required=True, help="③ predictions (checked facts)")
    s.add_argument("--none", help="① predictions (no photo information), optional")
    t = sub.add_parser("tally")
    t.add_argument("sheet")
    args = parser.parse_args()

    if args.command == "build":
        result = build(load_facts(args.facts), photo_of_reviews(args.reviews), args.allow_unchecked)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote facts for {len(result['predictions'])} reviews to {out}")
    elif args.command == "sheet":
        columns, rows = sheet_rows(by_review(args.vision), by_review(args.facts_run), by_review(args.none))
        sys.stdout.reconfigure(encoding="utf-8-sig", newline="")
        writer = csv.DictWriter(sys.stdout, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    else:
        print_tally(tally(read_sheet(args.sheet)))


if __name__ == "__main__":
    main()
