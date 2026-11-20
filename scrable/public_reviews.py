"""Turn a public Korean review CSV (e.g. the Mendeley delivery-app reviews, CC BY 4.0) into SCRABLE
review JSONL: training reviews from some restaurants, test reviews from other restaurants.

    python -m scrable.public_reviews inspect "data/raw/Korean Reviews.csv"
    python -m scrable.public_reviews build "data/raw/Korean Reviews.csv" --text-col <col> --store-col <col> \
        [--rating-col <col>] --out-train data/public_train.jsonl --out-test data/public_test.jsonl

Reviews are cleaned (length, Korean text, no phone numbers / e-mails / URLs / order numbers),
deduplicated over the whole file before splitting, and split by restaurant so no restaurant is in
both files. Restaurants get new ids (pub-001, ...); the id map stays in data/ (not in git).
Real restaurant names are never written to the review files.
"""

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from .data import normalize

MIN_CHARS, MAX_CHARS = 8, 300
SHORT_CHARS = 25  # Reviews shorter than this are capped so the set is not mostly "맛있어요".
MAX_SHORT_SHARE = 0.3

PII_PATTERNS = [
    re.compile(r"0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}"),  # phone numbers
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),  # e-mail
    re.compile(r"https?://|www\.", re.IGNORECASE),
    re.compile(r"주문\s*번호|\d{6,}"),  # order numbers, long digit runs
]
HANGUL = re.compile(r"[가-힣]")
LETTER = re.compile(r"[가-힣A-Za-z]")


def read_csv(path):
    """Rows as dicts. Tries UTF-8 (with or without BOM), then CP949."""
    for encoding in ("utf-8-sig", "cp949"):
        try:
            with open(path, encoding=encoding, newline="") as stream:
                return list(csv.DictReader(stream))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot read {path} as UTF-8 or CP949")


def clean(text):
    return " ".join(str(text or "").split())


def keep(text):
    """True when the review is usable: length, mostly Korean, no personal data."""
    if not MIN_CHARS <= len(text) <= MAX_CHARS:
        return False
    letters = LETTER.findall(text)
    if not letters or len(HANGUL.findall(text)) / len(letters) < 0.5:
        return False
    return not any(p.search(text) for p in PII_PATTERNS)


def parse_rating(value):
    try:
        rating = int(float(str(value).strip()))
    except ValueError:
        return None
    return rating if 1 <= rating <= 5 else None


def inspect(path, limit=3):
    rows = read_csv(path)
    print(f"{len(rows)} rows; columns: {list(rows[0]) if rows else []}")
    for column in rows[0] if rows else []:
        values = [clean(r[column]) for r in rows]
        distinct = Counter(values)
        mean_len = sum(map(len, values)) / len(values)
        print(f"\n[{column}] distinct={len(distinct)} mean_chars={mean_len:.0f}")
        if len(distinct) <= 12:
            print("  values:", dict(distinct.most_common(12)))
        else:
            for value in values[:limit]:
                print("  e.g.", value[:80])
    print(
        "\nPick --text-col (the review text: long, many distinct values), --store-col "
        "(the restaurant: few distinct values) and, if there is one, --rating-col (1-5 stars)."
    )


def sample_store(rows, count, rng, ratings):
    """count reviews from one restaurant: non-5-star first (up to half), short ones capped."""
    rows = rows[:]
    rng.shuffle(rows)
    if ratings:
        low = [r for r in rows if r["rating"] is not None and r["rating"] < 5][: count // 2]
        rows = low + [r for r in rows if r not in low]
    picked, short = [], 0
    for row in rows:
        if len(picked) == count:
            break
        if len(row["review"]) < SHORT_CHARS:
            if short >= int(count * MAX_SHORT_SHARE):
                continue
            short += 1
        picked.append(row)
    return picked


def build(
    rows,
    text_col,
    store_col,
    rating_col=None,
    train_stores=20,
    train_per_store=15,
    test_stores=10,
    test_per_store=4,
    seed=42,
):
    """(train rows, test rows, store id map, stats)."""
    for column in filter(None, (text_col, store_col, rating_col)):
        if rows and column not in rows[0]:
            raise ValueError(f"Column {column!r} not in CSV; run `inspect` to see the columns")
    by_store, seen, dropped = {}, set(), Counter()
    for raw in rows:
        text, store = clean(raw[text_col]), clean(raw[store_col])
        if not store:
            dropped["no_store"] += 1
            continue
        if not keep(text):
            dropped["filtered"] += 1
            continue
        key = normalize(text)
        if key in seen:  # Dedupe over the whole file before the split.
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        rating = parse_rating(raw[rating_col]) if rating_col else None
        by_store.setdefault(store, []).append({"review": text, "rating": rating})

    ratings = rating_col is not None
    rng = random.Random(seed)
    need = max(train_per_store, test_per_store)
    stores = sorted(s for s, items in by_store.items() if len(items) >= need)
    rng.shuffle(stores)
    if len(stores) < train_stores + test_stores:
        raise ValueError(
            f"Only {len(stores)} restaurants have {need}+ usable reviews; "
            f"need {train_stores + test_stores}. Lower the store or per-store counts."
        )
    test_names, train_names = stores[:test_stores], stores[test_stores : test_stores + train_stores]
    # Test restaurants are picked and sampled first, with their own id prefix, so asking for more
    # training restaurants later (e.g. 1000 reviews) keeps the same test set and the same first
    # training restaurants (their cached answers and judgments are reused).
    store_ids = {name: f"pubtest-{i:03d}" for i, name in enumerate(test_names, 1)}
    store_ids |= {name: f"pub-{i:03d}" for i, name in enumerate(train_names, 1)}

    def rows_for(names, per_store, kind):
        out = []
        for name in names:
            sid = store_ids[name]
            for n, item in enumerate(sample_store(by_store[name], per_store, rng, ratings), 1):
                out.append(
                    {
                        "id": f"{sid}-{kind}-{n:03d}",
                        "app_id": sid,
                        "rating": item["rating"],
                        "review": item["review"],
                        "expert_response": None,
                    }
                )
        return out

    test = rows_for(test_names, test_per_store, "test")
    train = rows_for(train_names, train_per_store, "train")
    stats = {
        "csv_rows": len(rows),
        "dropped": dict(dropped),
        "usable_restaurants": len(stores),
        "train": {
            "restaurants": len(train_names),
            "reviews": len(train),
            "ratings": dict(Counter(r["rating"] for r in train)),
        },
        "test": {
            "restaurants": len(test_names),
            "reviews": len(test),
            "ratings": dict(Counter(r["rating"] for r in test)),
        },
    }
    return train, test, store_ids, stats


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Public review CSV -> SCRABLE review JSONL")
    sub = parser.add_subparsers(dest="step", required=True)
    p_inspect = sub.add_parser("inspect", help="Show columns and examples")
    p_inspect.add_argument("csv")
    p_build = sub.add_parser("build", help="Clean, dedupe and split by restaurant")
    p_build.add_argument("csv")
    p_build.add_argument("--text-col", required=True)
    p_build.add_argument("--store-col", required=True)
    p_build.add_argument("--rating-col")
    p_build.add_argument("--train-stores", type=int, default=20)
    p_build.add_argument("--train-per-store", type=int, default=15)
    p_build.add_argument("--test-stores", type=int, default=10)
    p_build.add_argument("--test-per-store", type=int, default=4)
    p_build.add_argument("--out-train", default="data/public_train.jsonl")
    p_build.add_argument("--out-test", default="data/public_test.jsonl")
    p_build.add_argument("--map", default="data/public_store_map.json")
    p_build.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.step == "inspect":
        inspect(args.csv)
        return
    train, test, store_ids, stats = build(
        read_csv(args.csv),
        args.text_col,
        args.store_col,
        args.rating_col,
        args.train_stores,
        args.train_per_store,
        args.test_stores,
        args.test_per_store,
        args.seed,
    )
    write_jsonl(args.out_train, train)
    write_jsonl(args.out_test, test)
    Path(args.map).write_text(json.dumps(store_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(
        f"Wrote {len(train)} train reviews to {args.out_train}, {len(test)} test reviews to {args.out_test}"
    )


if __name__ == "__main__":
    main()
