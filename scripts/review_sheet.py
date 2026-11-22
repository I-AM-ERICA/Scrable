"""CSV sheet for a person to check replies for serious errors before a model switch.

    python scripts/review_sheet.py outputs/exp7b/photo_image/all_predictions.json > review_c.csv

One row per reply: review, photo file, photo description (if any), the reply, its judge score,
and empty columns to fill in: 중대 오류 (예/아니오) and 메모. A serious error is something an owner
must not post as is: a menu item or fact that neither the photo nor the review shows, a refund or
compensation promise without a store-document basis, blaming the customer. Opens in Excel
(UTF-8 with BOM).
"""

import csv
import json
import sys
from pathlib import Path

COLUMNS = ["review_id", "review", "image", "photo_description", "reply", "score", "중대 오류", "메모"]


def rows(predictions_path):
    data = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    for p in data["predictions"]:
        yield {
            "review_id": p["review_id"],
            "review": p["review"],
            "image": p.get("image") or "",
            "photo_description": p.get("photo_description") or "",
            "reply": p["prediction"],
            "score": round(p["score"], 3),
            "중대 오류": "",
            "메모": "",
        }


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.stdout.reconfigure(encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows(sys.argv[1]))


if __name__ == "__main__":
    main()
