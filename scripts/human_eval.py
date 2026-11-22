"""Blind human evaluation of test-split answers from several runs.

Build the rating page (answers shuffled into slots A/B/C, the slot->run key kept apart):

    python scripts/human_eval.py build \
        --run stage1:optimized=outputs/stage1/test_predictions.json \
        --run dpo=outputs/stage2_eval/test_predictions.json \
        --run claude-haiku-5-5:base=outputs/stage1/test_predictions.json

Score collected ratings (one JSON per rater: the page's "결과 복사" text, or a db export):

    python scripts/human_eval.py score outputs/human_eval/ratings/*.json

Google Forms instead of the page (pick the best reply per review; responses land in a sheet):

    python scripts/human_eval.py form --dir outputs/human_eval_short
    # paste make_google_form.gs into script.google.com, run makeForm, share the link
    python scripts/human_eval.py score --csv responses.csv --key outputs/human_eval_short/blind_key.json

The run name before '=' is a key of scrable.evaluate.load_runs for that file.
Raters see only the page; blind_key.json never goes into it.
"""

import argparse
import csv
import glob
import itertools
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scrable.evaluate import load_runs, paired_bootstrap  # noqa: E402
from scrable.io import write_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts" / "human_eval_page.html"
SLOTS = "ABCDE"


def store_title(markdown, app_id):
    match = re.search(r"^#\s*(.+?)(?:\s*가게 정보)?\s*$", markdown, re.M)
    return match.group(1) if match else app_id


def pick_subset(ids, run, ratings, size, seed):
    """`size` reviews spread evenly over stores; within a store half complaints (1-3 stars), half praise.

    Praise reviews are easy for every model, so an even split keeps the subset informative.
    """
    by_store = {}
    for rid in ids:
        by_store.setdefault(run[rid]["app_id"], []).append(rid)
    stores = sorted(by_store)
    rng = random.Random(f"{seed}:subset")
    picked = []
    for i, store in enumerate(stores):
        want = size // len(stores) + (i < size % len(stores))
        pool = sorted(by_store[store])
        rng.shuffle(pool)
        low = [r for r in pool if (ratings.get(r) or 5) <= 3]
        high = [r for r in pool if r not in low]
        take_low = min(len(low), want - min(len(high), want // 2))
        picked += low[:take_low] + high[: want - take_low]
    return sorted(picked)


def build(args):
    runs = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        available = load_runs(path)
        if name not in available:
            raise SystemExit(f"{path} has no run {name!r}; it has {sorted(available)}")
        runs[name] = available[name]
    if not 2 <= len(runs) <= len(SLOTS):
        raise SystemExit(f"Give between 2 and {len(SLOTS)} --run values")
    ids = sorted(next(iter(runs.values())))
    for name, run in runs.items():
        if sorted(run) != ids:
            raise SystemExit(f"Run {name} covers different test reviews")

    ratings = {}
    if Path(args.reviews).exists():
        for line in Path(args.reviews).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                ratings[row["id"]] = row.get("rating")
    if args.subset:
        ids = pick_subset(ids, next(iter(runs.values())), ratings, args.subset, args.seed)
    stores = {}
    items, key = [], {}
    # Every ordering used equally often (shuffled), so no run sits in slot A more than the others.
    # pair mode shows two answers per review: each ordered pair of runs (3 runs -> 6 cases) equally often.
    perms = list(itertools.permutations(runs, 2 if args.mode == "pair" else len(runs)))
    if args.mode == "pair" and len(ids) % len(perms):
        raise SystemExit(f"pair mode: use a multiple of {len(perms)} reviews (--subset) so every case appears equally")
    random.Random(args.seed).shuffle(perms)
    orders = [list(perms[i % len(perms)]) for i in range(len(ids))]
    random.Random(f"{args.seed}:orders").shuffle(orders)
    for review_id, names in zip(ids, orders):
        first = next(iter(runs.values()))[review_id]
        app_id = first["app_id"]
        if app_id not in stores:
            info = Path(args.docs) / app_id / "info.md"
            text = info.read_text(encoding="utf-8") if info.exists() else ""
            stores[app_id] = {"name": store_title(text, app_id), "info": text}
        slots = dict(zip(SLOTS, names))
        key[review_id] = slots
        items.append(
            {
                "id": review_id,
                "app_id": app_id,
                "rating": ratings.get(review_id),
                "review": first["review"],
                "answers": [
                    {"slot": slot, "text": runs[name][review_id]["prediction"]}
                    for slot, name in slots.items()
                ],
            }
        )

    out = Path(args.out_dir)
    payload = {"mode": args.mode, "stores": stores, "items": items}
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    page = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/{}", data)
    if args.mode == "rank":
        page = page.replace("<title>리뷰 답변 블라인드 평가</title>", "<title>리뷰 답변 고르기</title>")
    out.mkdir(parents=True, exist_ok=True)
    if args.mode != "pair":  # pair mode is collected with the Google Form only
        (out / "human_eval.html").write_text(page, encoding="utf-8")
    write_json(out / "items.json", payload)
    write_json(
        out / "blind_key.json", {"runs": list(runs), "seed": args.seed, "mode": args.mode, "slots": key}
    )
    width = len(next(iter(key.values())))
    counts = {n: [sum(k.get(s) == n for k in key.values()) for s in SLOTS[:width]] for n in runs}
    print(f"{len(items)} reviews, runs {list(runs)}")
    print("times in slot " + "/".join(SLOTS[:width]) + ": " + json.dumps(counts))
    print(f"Wrote files to {out} (keep blind_key.json away from raters)")


def one_line(text):
    return re.sub(r"\s+", " ", text).strip()


FORM_SCRIPT = """// Paste into https://script.google.com (new project), choose makeForm, press Run.
// It creates the Google Form and its response sheet in your Drive and logs the links.
var TITLE = __TITLE__;
var DESC = __DESC__;
var DATA = __DATA__;
var ASK_NAME = __ASK_NAME__;

function makeForm() {
  var form = FormApp.create(TITLE);
  form.setDescription(DESC);
  form.setProgressBar(true);
  form.setShuffleQuestions(false);
  if (ASK_NAME) {
    form.addTextItem().setTitle('이름 또는 별명 (선택)')
        .setHelpText('같은 사람이 두 번 냈는지 확인하는 데만 씁니다.');
  }
  DATA.forEach(function (store) {
    if (store.info) form.addPageBreakItem().setTitle(store.name + ' (가게 정보는 필요할 때만 보세요)').setHelpText(store.info);
    store.items.forEach(function (it) {
      form.addMultipleChoiceItem()
          .setTitle(it.title)
          .setHelpText(it.help)
          .setChoiceValues(it.choices)
          .setRequired(true);
    });
  });
  var sheet = SpreadsheetApp.create(TITLE + ' (응답)');
  form.setDestination(FormApp.DestinationType.SPREADSHEET, sheet.getId());
  Logger.log('보낼 링크: ' + form.getPublishedUrl());
  Logger.log('설문 편집: ' + form.getEditUrl());
  Logger.log('응답 시트: ' + sheet.getUrl());
}
"""

PAIR_DESC = "손님 리뷰마다 사장님 답글이 2개 있어요. 더 나은 쪽 하나만 누르면 됩니다. ({n}문항, 약 {minutes}분)"

FORM_DESC = (
    "가게 리뷰에 AI가 단 사장님 답글을 비교하는 설문입니다. "
    "리뷰마다 답글 후보가 3개 있고, 내가 사장님이라면 올리고 싶은 답글 하나만 고르면 됩니다. "
    "어떤 AI가 썼는지는 숨겼습니다. 5~10분 걸립니다.\n\n"
    "고를 때 볼 것: ① 손님이 한 말에 맞게 답했나 ② 가게 정보와 맞나 "
    "(확인도 안 하고 환불·재배달을 약속하면 안 됩니다) ③ 말투가 자연스럽고 정중한가. "
    "길이는 따지지 마세요. 가게 정보는 각 페이지 맨 위에 있습니다. 모든 가게는 실험용 가상 매장입니다."
)


def sentence_lines(text):
    """One sentence per line, for reading on a phone."""
    text = re.sub(r"[ \t]+", " ", text.strip())
    return re.sub(r"(?<=[.!?])\s+", "\n", text)


def form(args):
    """Write a Google Apps Script that builds the form, plus the question -> review map."""
    out = Path(args.dir)
    data = json.loads((out / "items.json").read_text(encoding="utf-8"))
    stores, number, qmap = {}, 0, {}
    for item in sorted(data["items"], key=lambda i: (i["app_id"], i["id"])):
        number += 1
        store = data["stores"].get(item["app_id"], {"name": item["app_id"], "info": ""})
        info = re.sub(r"^#.*\n+", "", store["info"]).replace("## ", "■ ").strip()
        entry = stores.setdefault(item["app_id"], {"name": store["name"], "info": info, "items": []})
        stars = "★" * item["rating"] + "☆" * (5 - item["rating"]) if item.get("rating") else ""
        title = f"Q{number:02d}. {stars} “{one_line(item['review'])}”"
        if data.get("mode") == "pair":
            # Self-explanatory question: the review and both replies sit in the question itself.
            first, second = item["answers"]
            title = f"Q{number:02d}. 내가 사장님이라면 어느 답글을 올릴까요?"
            help_text = (
                f"손님 리뷰 {stars}\n“{one_line(item['review'])}”\n\n"
                "① 첫 번째 답글\n" + sentence_lines(first["text"])
                + "\n\n② 두 번째 답글\n" + sentence_lines(second["text"])
            )
            entry["info"] = ""  # no store-info pages: nothing to read before answering
            choices = list(PAIR_CHOICES)
        else:
            help_text = "사장님 답글로 가장 좋은 것 하나를 골라 주세요."
            choices = [one_line(a["text"]) for a in item["answers"]]
            if len(set(choices)) != len(choices):
                raise SystemExit(f"{item['id']} has identical answers; a form cannot tell them apart")
        entry["items"].append({"title": title, "help": help_text, "choices": choices})
        qmap[f"Q{number:02d}"] = item["id"]
    pair = data.get("mode") == "pair"
    chars = sum(len(i["review"]) + sum(len(a["text"]) for a in i["answers"]) for i in data["items"])
    minutes = max(1, round(chars / 500 + number * 4 / 60))
    description = PAIR_DESC.format(n=number, minutes=minutes) if pair else FORM_DESC
    script = (
        FORM_SCRIPT.replace("__TITLE__", json.dumps(args.title, ensure_ascii=False))
        .replace("__DESC__", json.dumps(description, ensure_ascii=False))
        .replace("__ASK_NAME__", "false" if pair else "true")
        .replace("__DATA__", json.dumps(list(stores.values()), ensure_ascii=False, indent=1))
    )
    (out / "make_google_form.gs").write_text(script, encoding="utf-8")
    write_json(out / "form_map.json", qmap)
    print(f"{number} questions. Wrote {out / 'make_google_form.gs'} and {out / 'form_map.json'}")


PAIR_CHOICES = ("① 첫 번째 답글", "② 두 번째 답글")


def load_form_csv(path, items_path, map_path):
    """Raters from a Google Forms response sheet saved as CSV (one row per person)."""
    items = {i["id"]: i for i in json.loads(Path(items_path).read_text(encoding="utf-8"))["items"]}
    qmap = json.loads(Path(map_path).read_text(encoding="utf-8"))
    raters = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row_number, row in enumerate(csv.DictReader(f), 1):
            name, picks = "", {}
            for header, value in row.items():
                match = re.match(r"\s*(Q\d+)\.", header or "")
                if not match:
                    if header and "이름" in header:
                        name = (value or "").strip()
                    continue
                rid = qmap.get(match.group(1))
                by_text = {one_line(a["text"]): a["slot"] for a in items[rid]["answers"]}
                if len(items[rid]["answers"]) == 2:  # pair mode: choices are the two labels
                    by_text.update(dict(zip(PAIR_CHOICES, (a["slot"] for a in items[rid]["answers"]))))
                slot = by_text.get(one_line(value or ""))
                if slot:
                    picks[rid] = {"best": slot}
            raters.append({"name": f"{row_number}:{name}" if name else f"row{row_number}", "items": picks})
    return raters


def load_ratings(paths):
    raters = []
    # PowerShell passes "*.json" through unexpanded, so expand patterns here.
    paths = [p for pattern in paths for p in (sorted(glob.glob(pattern)) or [pattern])]
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # A db export may wrap the document body.
        data = data.get("data", data) if "items" not in data else data
        raters.append({"name": data.get("rater") or Path(path).stem, "items": data.get("items", {})})
    return raters


def rater_points(answers, slots, mode):
    """{run: points} for one rater on one review, or None if incomplete.

    score mode: the 1-5 score. rank mode: best 1, worst 0, the remaining answer 0.5.
    """
    if mode in ("rank", "pair"):
        best, worst = answers.get("best"), answers.get("worst")
        if best in slots and worst is None:  # best-only (Google Forms): 1 for the pick, 0 otherwise
            return {run: 1.0 if s == best else 0.0 for s, run in slots.items()}
        if best not in slots or worst not in slots or best == worst:
            return None
        return {run: 1.0 if s == best else 0.0 if s == worst else 0.5 for s, run in slots.items()}
    if not all(isinstance(answers.get(s, {}).get("score"), int) for s in slots):
        return None
    return {run: float(answers[s]["score"]) for s, run in slots.items()}


def sign(x):
    return (x > 0) - (x < 0)


def wilson(wins, n, z=1.96):
    """95% Wilson interval for a win rate."""
    if not n:
        return 0.0, 1.0
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return centre - half, centre + half


def score_pairs(per, runs, slots, judged):
    """Pair mode: each review compares two runs; report head-to-head win rates."""
    total = sum(len(v) for v in per.values())
    raters = {name for v in per.values() for name in v}
    print(f"\n{len(per)} reviews, {len(raters)} raters, {total} choices (each choice: which of two replies is better)")
    first = sum(points[slots[rid]["A"]] for rid, v in per.items() for points in v.values())
    print(f"First-shown reply picked {first:.0f}/{total} ({first / total:.0%}; near 50% means no position bias)")

    print("\nHead to head (95% Wilson interval of the first run's win rate; 'clear' if it excludes 50%)")
    for a, b in itertools.combinations(runs, 2):
        both = [rid for rid in per if set(slots[rid].values()) == {a, b}]
        wins = sum(points[a] for rid in both for points in per[rid].values())
        n = sum(len(per[rid]) for rid in both)
        if not n:
            print(f"  {a} vs {b}: no answers yet")
            continue
        low, high = wilson(wins, n)
        verdict = "chance?" if low <= 0.5 <= high else "clear"
        print(
            f"  {a} beats {b}: {wins:.0f}/{n} = {wins / n:.0%} [{low:.0%}, {high:.0%}]"
            f"  ({len(both)} reviews)  {verdict}"
        )

    print("\nOverall win rate against the other two")
    for run in runs:
        mine = [points[run] for rid, v in per.items() if run in slots[rid].values() for points in v.values()]
        print(f"  {run:<24} {sum(mine):.0f}/{len(mine)} = {sum(mine) / len(mine):.0%}")

    agree = pairs = 0
    for v in per.values():
        for x, y in itertools.combinations(v.values(), 2):
            pairs += 1
            agree += x == y
    if pairs:
        print(f"\nTwo raters picked the same reply in {agree}/{pairs} rater pairs ({agree / pairs:.0%}; 50% = chance)")

    if len(judged) == len(runs):
        same = n = 0
        for rid, v in per.items():
            a, b = slots[rid].values()
            human = sign(sum(p[a] - p[b] for p in v.values()))
            judge = sign(judged[a][rid]["score"] - judged[b][rid]["score"])
            if human and judge:
                n += 1
                same += human == judge
        if n:
            print(
                f"LLM judge prefers the same reply as the raters' majority in {same}/{n} reviews "
                f"({same / n:.0%}; reviews with a tie on either side skipped)"
            )


def score(args):
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    runs, slots, mode = key["runs"], key["slots"], key.get("mode", "score")
    raters = load_ratings(args.ratings) if args.ratings else []
    key_dir = Path(args.key).parent
    for path in args.csv:
        raters += load_form_csv(path, key_dir / "items.json", key_dir / "form_map.json")
    if not raters:
        raise SystemExit("Give rating JSON files and/or --csv")
    # per[review_id][rater] = {run: points}; flags[run] = list of bools (score mode)
    per, flags = {}, {run: [] for run in runs}
    for rater in raters:
        done = 0
        for review_id, answers in rater["items"].items():
            if review_id not in slots:
                continue
            points = rater_points(answers, slots[review_id], mode)
            if points is None:
                continue
            done += 1
            per.setdefault(review_id, {})[rater["name"]] = points
            if mode == "score":
                for s, run in slots[review_id].items():
                    flags[run].append(bool(answers[s].get("flag")))
        print(f"rater {rater['name']}: {done}/{len(slots)} reviews complete")
    common = sorted(per)
    if not common:
        raise SystemExit("No review has complete ratings yet")

    def mean_points(run, rid):
        rows = per[rid].values()
        return sum(r[run] for r in rows) / len(rows)

    if mode == "pair":
        score_pairs(per, runs, slots, {run: load_runs(path)[run] for run, path in args.judge.items()})
        return

    total = sum(len(v) for v in per.values())
    if mode == "rank":
        print(f"\n{len(common)} reviews, {total} rankings; points: best 1, middle 0.5, worst 0 (best-only form: best 1, others 0)")
        three_way = any(r[run] == 0.5 for v in per.values() for r in v.values() for run in runs)
        for run in runs:
            rows = [r[run] for v in per.values() for r in v.values()]
            worst = f"  worst {rows.count(0.0) / len(rows):.0%}" if three_way else ""
            print(f"  {run:<24} mean {sum(rows) / len(rows):.2f}  picked best {rows.count(1.0) / len(rows):.0%}{worst}")
    else:
        print(f"\n{len(common)} reviews, {total} rater-reviews; score 1-5; flagged = fact error or unconfirmed promise")
        for run in runs:
            rows = [r[run] for v in per.values() for r in v.values()]
            print(
                f"  {run:<24} mean {sum(rows) / len(rows):.2f}  score>=4 {sum(x >= 4 for x in rows) / len(rows):.0%}"
                f"  flagged {sum(flags[run]) / len(flags[run]):.0%}"
            )

    print("\nPaired differences per review (mean over raters; 95% bootstrap CI; win/lose/tie)")
    for a, b in itertools.combinations(runs, 2):
        r = paired_bootstrap([mean_points(a, i) for i in common], [mean_points(b, i) for i in common])
        verdict = "chance?" if r["low"] <= 0 <= r["high"] else "clear"
        print(
            f"  {a} - {b}: {r['mean']:+.2f} [{r['low']:+.2f}, {r['high']:+.2f}] "
            f"{r['wins']}/{r['losses']}/{r['ties']}  {verdict}"
        )

    agree = pairs = 0
    for rid in common:
        for x, y in itertools.combinations(per[rid].values(), 2):
            for a, b in itertools.combinations(runs, 2):
                pairs += 1
                agree += sign(x[a] - x[b]) == sign(y[a] - y[b])
    if pairs:
        print(f"\nRater agreement on which of two answers is better (ties count as their own answer): {agree / pairs:.0%} of {pairs} comparisons")

    judged = {run: load_runs(path)[run] for run, path in args.judge.items()}
    if judged:
        print("\nLLM judge on the same reviews (normalized 0-1):")
        for run, fb in judged.items():
            print(f"  {run:<24} {sum(fb[i]['score'] for i in common) / len(common):.3f}")
        same = n = 0
        for rid in common:
            for a, b in itertools.combinations(judged, 2):
                human = sign(mean_points(a, rid) - mean_points(b, rid))
                judge = sign(judged[a][rid]["score"] - judged[b][rid]["score"])
                if human and judge:
                    n += 1
                    same += human == judge
        if n:
            print(f"Judge picks the same better answer as the raters' average in {same}/{n} pairs ({same / n:.0%}; ties skipped)")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--run", action="append", required=True, help="name=path/to/test_predictions.json")
    b.add_argument("--reviews", default="data/store_reviews.jsonl")
    b.add_argument("--docs", default="data/store_docs")
    b.add_argument("--out-dir", default="outputs/human_eval")
    b.add_argument("--seed", default="human-eval-1")
    b.add_argument("--mode", choices=["score", "rank", "pair"], default="score", help="1-5 scores, pick best/worst of all, or pick the better of two")
    b.add_argument("--subset", type=int, help="Only this many reviews, spread over stores and star ratings")
    f = sub.add_parser("form")
    f.add_argument("--dir", default="outputs/human_eval_short", help="A build output folder (needs items.json)")
    f.add_argument("--title", default="가게 리뷰 답글 고르기")
    s = sub.add_parser("score")
    s.add_argument("ratings", nargs="*")
    s.add_argument("--csv", action="append", default=[], help="Google Forms responses saved as CSV")
    s.add_argument("--key", default="outputs/human_eval/blind_key.json")
    s.add_argument("--judge", action="append", default=[], help="name=path to also print the LLM judge score")
    args = parser.parse_args()
    if args.command == "form":
        form(args)
    elif args.command == "build":
        if args.mode == "rank" and len(args.run) != 3:
            raise SystemExit("rank mode needs exactly 3 --run values")
        build(args)
    else:
        args.judge = dict(spec.split("=", 1) for spec in args.judge)
        score(args)


if __name__ == "__main__":
    main()
