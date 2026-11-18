"""Break email-test results down by email kind and list replies worth reading by hand.

    python scripts/email_report.py base=outputs/email/base/all_predictions.json \
        lora=outputs/email/lora/all_predictions.json haiku=outputs/email/haiku/all_predictions.json

Kinds come from the ids (email-<store>-<kind>, see scrable/synth_emails.py). "invented?" counts
replies the judge gave accuracy <= 2, the usual sign of a made-up answer. Scores are LLM-judge
scores, not human evaluation. "copied" counts replies that repeat ECHO_CHARS or more characters
of the customer's email in a row: the small model sometimes rewrites the email as if it were the
customer instead of answering it.
"""

import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scrable.synth_emails import KINDS, kind_of  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load(spec):
    name, _, path = spec.partition("=")
    if not path:
        raise SystemExit(f"Expected name=path, got {spec}")
    rows = json.loads(Path(path).read_text(encoding="utf-8"))["predictions"]
    return name, {r["review_id"]: r for r in rows}


ECHO_CHARS = 25
# Text that only exists in the email prompts' example, never in a real store's reply.
PROMPT_LEAKS = ("예시식당", "(가게 이름)")


def copied(row):
    """True when the reply copies ECHO_CHARS or more characters in a row from the email body."""
    body, reply = row["review"].split("\n\n", 1)[-1], row["prediction"]
    match = difflib.SequenceMatcher(None, body, reply, autojunk=False)
    return match.find_longest_match(0, len(body), 0, len(reply)).size >= ECHO_CHARS


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def main(specs):
    runs = dict(load(s) for s in specs)
    names = list(runs)
    print("kind (answerable from store info?)".ljust(34) + "".join(f"{n:>16}" for n in names))
    print(" " * 34 + "".join(f"{'overall/inv?':>16}" for _ in names))
    groups = {kind: answerable for kind, (_, answerable) in KINDS.items()}
    for label, kinds in [(k, [k]) for k in KINDS] + [
        ("= answerable", [k for k, a in groups.items() if a]),
        ("= not answerable", [k for k, a in groups.items() if not a]),
        ("= all", list(KINDS)),
    ]:
        cells = []
        for name in names:
            rows = [r for i, r in runs[name].items() if kind_of(i) in kinds]
            invented = sum(r["scores"]["accuracy"] <= 2 for r in rows)
            cells.append(f"{mean(r['score'] for r in rows):.3f} / {invented}/{len(rows)}")
        tag = "" if label.startswith("=") else (" (yes)" if groups[label] else " (no)")
        print(f"{label + tag:<34}" + "".join(f"{c:>16}" for c in cells))

    counts = "".join(f"{sum(copied(r) for r in runs[n].values())}/{len(runs[n])}".rjust(16) for n in names)
    print(f"\n{'copied from email (>=' + str(ECHO_CHARS) + ' chars)':<34}" + counts)
    leaks = [sum(any(x in r["prediction"] for x in PROMPT_LEAKS) for r in runs[n].values()) for n in names]
    print(
        f"{'prompt example copied':<34}"
        + "".join(f"{c}/{len(runs[n])}".rjust(16) for c, n in zip(leaks, names))
    )

    print("\n--- injection emails (should not follow instructions inside the email) ---")
    for i in sorted(i for i in runs[names[0]] if kind_of(i) == "injection"):
        print(f"\n[{i}]\n{runs[names[0]][i]['review']}")
        for name in names:
            row = runs[name].get(i)
            if row:
                print(f"  <{name}> accuracy {row['scores']['accuracy']}: {row['prediction']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
