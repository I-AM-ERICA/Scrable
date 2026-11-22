"""Photo reviews for training: public photo-review text + food photos of the same menu.

    python -m scrable.public_photo_reviews menus "data/raw/Reviews Written in Korean.csv"
    python -m scrable.public_photo_reviews build "data/raw/Reviews Written in Korean.csv" \
        --photos data/raw/food_photos --out data/public_photo_reviews.jsonl

The Mendeley CSV marks reviews that had a photo (HasPicture=1) and lists the ordered menu, but the
photos themselves are not distributed. Food photos with a food-name label (e.g. AI Hub "한국 이미지
(음식)", one folder per food) are matched to those reviews by menu name: a 떡볶이 photo goes to a
review whose order contained 떡볶이. The photo is a stand-in for the customer's photo (same dish,
not the same plate), which is enough to teach the reply model to use photo_description.

`menus` counts matching photo reviews per food name (no photos needed) so you can choose which
food folders to download. `build` needs the photo folders.

Restaurants in the public test set are never used. Restaurants already in the text training set
keep their id (pub-NNN) and store document; new ones get pubimg-NNN and need store documents
(scrable.synth_store_docs, which keeps existing documents).
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from .data import normalize
from .photos import MEDIA_TYPES
from .public_reviews import clean, keep, parse_rating, read_csv, write_jsonl

# Common delivery dishes that also appear as AI Hub Korean-food classes. Used by `menus` when no
# --foods list is given; the real class list is whatever folders you download.
DEFAULT_FOODS = [
    "떡볶이",
    "김밥",
    "순대",
    "라면",
    "김치찌개",
    "된장찌개",
    "부대찌개",
    "순두부찌개",
    "제육볶음",
    "닭볶음탕",
    "감자탕",
    "갈비탕",
    "설렁탕",
    "육개장",
    "삼계탕",
    "짜장면",
    "짬뽕",
    "탕수육",
    "볶음밥",
    "김치볶음밥",
    "비빔밥",
    "불고기",
    "갈비찜",
    "찜닭",
    "닭갈비",
    "족발",
    "보쌈",
    "삼겹살",
    "곱창구이",
    "막창구이",
    "칼국수",
    "수제비",
    "물냉면",
    "비빔냉면",
    "만두",
    "잡채",
    "파전",
    "김치전",
    "해물찜",
    "아귀찜",
    "양념치킨",
    "후라이드치킨",
    "깐풍기",
    "주꾸미볶음",
    "오징어볶음",
    "낙지볶음",
    "콩나물국밥",
    "돼지국밥",
    "순대국",
    "떡국",
    "잔치국수",
    "비빔국수",
    "계란말이",
    "계란찜",
    "어묵볶음",
]  # Not "김치": it is usually a side dish, so the photo would show the wrong food.

COLS = {"text": "Review", "store": "RestaurantID", "rating": "Total", "menu": "Menu", "photo": "HasPicture"}


def squash(text):
    return "".join(str(text or "").split())


def menu_names(menu):
    """Ordered dish names only: options in parentheses and add-ons ending in 사리 are dropped,
    so "짜글이 정식/1(짜글이 선택(김치찌개 짜글이))" does not match 김치찌개."""
    text = str(menu or "").replace("（", "(").replace("）", ")")
    previous = None
    while previous != text:  # Remove nested parentheses from the inside out.
        previous, text = text, re.sub(r"\([^()]*\)", "", text)
    return re.sub(r"[가-힣]*사리", "", squash(text))


EVENT_ITEM = re.compile(r"이벤트|EVENT|VENT|후기|리뷰|찜&|서비스", re.IGNORECASE)
# A food name followed by one of these is a different dish: 순대국 is not 순대, 육회비빔밥 not 육회.
LONGER_DISH = ("국", "탕", "전골", "찌개", "볶음밥", "덮밥", "비빔밥", "국수", "라면")


def main_item(menu):
    """The first ordered dish (the main one): event/review-promise items are skipped, and side
    orders after it (drinks, 새우튀김 추가, 공기밥) are ignored."""
    for item in menu_names(menu).split(","):
        name = item.split("/")[0]
        if name and not EVENT_ITEM.search(name):
            return name
    return ""


def food_of(menu, foods):
    """Longest food name in the main dish (물냉면 wins over 냉면), or None."""
    item = main_item(menu)
    found = []
    for food in foods:
        key = squash(food)
        start = item.find(key)
        if start >= 0 and not item[start + len(key) :].startswith(LONGER_DISH):
            found.append(food)
    return max(found, key=len) if found else None


def names_other_food(text, food, foods):
    """True when the review talks about another dish (e.g. a 해물찜 review whose order list starts
    with 계란찜): the photo would not match what the customer wrote about."""
    text, key = squash(text), squash(food)
    return any(squash(f) in text and key not in squash(f) for f in foods if f != food)


def photo_rows(rows, foods, excluded_stores=(), avoid_texts=()):
    """Usable photo reviews with their matched food: [(row, food)]. Deduplicated by text."""
    seen = {normalize(t) for t in avoid_texts}
    out = []
    for raw in rows:
        if clean(raw.get(COLS["photo"])) != "1" or clean(raw[COLS["store"]]) in excluded_stores:
            continue
        text = clean(raw[COLS["text"]])
        if not keep(text) or normalize(text) in seen:
            continue
        food = food_of(raw.get(COLS["menu"]), foods)
        if food is None or names_other_food(text, food, set(foods) | set(DEFAULT_FOODS)):
            continue
        seen.add(normalize(text))
        out.append((raw, food))
    return out


def find_food_photos(directory):
    """{food name: [photo paths]} from folders that directly contain images (folder name = food)."""
    photos = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and path.suffix.lower() in MEDIA_TYPES:
            photos.setdefault(path.parent.name, []).append(path)
    if not photos:
        raise ValueError(f"No JPG/PNG/WEBP photos under {directory} (one folder per food)")
    return photos


def build(rows, photos, store_map, count=200, per_food=20, seed=42, avoid_texts=()):
    """(review rows with image paths, stats). store_map: RestaurantID -> pub-/pubtest- id."""
    excluded = {rid for rid, sid in store_map.items() if sid.startswith("pubtest-")}
    matched = photo_rows(rows, sorted(photos), excluded, avoid_texts)
    rng = random.Random(seed)
    by_food = {}
    for raw, food in matched:
        by_food.setdefault(food, []).append(raw)
    queues = {}
    for food, items in sorted(by_food.items()):
        rng.shuffle(items)
        low = [r for r in items if (parse_rating(r[COLS["rating"]]) or 5) < 5][: per_food // 3]
        queues[food] = (low + [r for r in items if r not in low])[:per_food]
    picked = []  # Round-robin over foods so no single dish dominates.
    while len(picked) < count and any(queues.values()):
        for food in sorted(queues):
            if queues[food] and len(picked) < count:
                picked.append((queues[food].pop(0), food))
    new_ids, used = {}, Counter()
    next_new = 1
    out = []
    for raw, food in picked:
        rid = clean(raw[COLS["store"]])
        sid = store_map.get(rid)
        if sid is None:
            if rid not in new_ids:
                new_ids[rid] = f"pubimg-{next_new:03d}"
                next_new += 1
            sid = new_ids[rid]
        pool = photos[food]
        image = pool[used[food] % len(pool)]  # Spread photos; reuse only when a food runs out.
        used[food] += 1
        out.append(
            {
                "id": f"{sid}-photo-{len(out) + 1:03d}",
                "app_id": sid,
                "rating": parse_rating(raw[COLS["rating"]]),
                "review": clean(raw[COLS["text"]]),
                "expert_response": None,
                "image": image.as_posix(),
            }
        )
    stats = {
        "photo_reviews_matched": len(matched),
        "picked": len(out),
        "by_food": dict(Counter(food for _, food in picked)),
        "foods_without_reviews": sorted(set(photos) - set(by_food)),
        "stores_reused": len({r["app_id"] for r in out if r["app_id"].startswith("pub-")}),
        "stores_new": len(new_ids),
        "ratings": dict(Counter(r["rating"] for r in out)),
    }
    return out, new_ids, stats


def main():
    parser = argparse.ArgumentParser(description="Public photo-review text + food photos by menu")
    sub = parser.add_subparsers(dest="step", required=True)
    p_menus = sub.add_parser("menus", help="Count photo reviews per food name (no photos needed)")
    p_menus.add_argument("csv")
    p_menus.add_argument("--foods", help="Comma-separated food names (default: common delivery dishes)")
    p_menus.add_argument("--store-map", default="data/public_store_map.json")
    p_build = sub.add_parser("build", help="Write photo reviews with matched photos")
    p_build.add_argument("csv")
    p_build.add_argument("--photos", required=True, help="Folder with one subfolder of photos per food")
    p_build.add_argument("--store-map", default="data/public_store_map.json")
    p_build.add_argument("--avoid", nargs="*", default=["data/public_train.jsonl", "data/public_test.jsonl"])
    p_build.add_argument("--out", default="data/public_photo_reviews.jsonl")
    p_build.add_argument("--count", type=int, default=200)
    p_build.add_argument("--per-food", type=int, default=20)
    p_build.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    store_map = json.loads(Path(args.store_map).read_text(encoding="utf-8"))
    rows = read_csv(args.csv)
    if args.step == "menus":
        foods = [f.strip() for f in args.foods.split(",")] if args.foods else DEFAULT_FOODS
        excluded = {rid for rid, sid in store_map.items() if sid.startswith("pubtest-")}
        counts = Counter(food for _, food in photo_rows(rows, foods, excluded))
        for food, n in counts.most_common():
            print(f"{n:6d}  {food}")
        print(f"\n{len(counts)} of {len(foods)} foods have photo reviews; download photos for the top ones.")
        return
    avoid = []
    for path in args.avoid:
        if Path(path).exists():
            avoid += [
                json.loads(line)["review"]
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    out, new_ids, stats = build(
        rows, find_food_photos(args.photos), store_map, args.count, args.per_food, args.seed, avoid
    )
    write_jsonl(args.out, out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Wrote {len(out)} photo reviews to {args.out}; {len(new_ids)} new stores need store documents")


if __name__ == "__main__":
    main()
