"""Write one fictional store-info document per restaurant in a review JSONL, with an API model.

    python -m scrable.synth_store_docs --reviews data/public_train.jsonl --out data/public_train_docs

Public review data has reviews but no store information, and SCRABLE answers from store documents.
So for each restaurant (app_id) the model reads up to MAX_REVIEWS of its reviews and invents a store
document in the same layout as data/train_store_docs: the menu items the reviews mention, hours,
delivery and contact routes, and a policy for problems. Names are invented; the document says it is
fictional. Policies are varied on purpose (POLICY_STYLES) so the reply model learns to read each
store's own policy. Requests are cached, so a rerun costs nothing.
"""

import argparse
import json
import random
from pathlib import Path

from .config import Config
from .data import load_custom_reviews
from .llm import CachedLLM, make_backend, structured_call

MAX_REVIEWS = 20

POLICY_STYLES = [
    "누락 메뉴는 확인되면 다시 보내 드리고, 음식 상태 문제는 확인 후 개별 안내",
    "누락·오배송은 배달앱 고객센터로 접수하면 환불 처리, 재배달은 하지 않음",
    "문제가 있으면 매장 전화로 연락받아 확인 후 재조리 여부를 안내",
    "보상 정책을 따로 정하지 않고, 모든 문제는 확인 후 개별 안내",
]
EVENT_STYLES = [
    "주문 시 리뷰 작성을 약속하면 음료나 사이드 서비스 (별점·내용은 자유)",
    "리뷰 이벤트 없음",
]

SYSTEM = """당신은 리뷰 답글 연구용 가상 가게 정보를 작성합니다.
주어진 실제 고객 리뷰들을 읽고, 그 리뷰와 모순되지 않는 가상의 가게 정보 문서를 마크다운으로 작성하세요.
- 가게 이름은 새로 지어내세요. 리뷰에 나온 가게 이름이나 실제 상호, 사람 이름, 전화번호, 주소는 쓰지 마세요.
- 첫 줄은 "# <가게 이름> 가게 정보", 그다음 줄은 "※ 실험용 가상 매장입니다. 공개 리뷰 데이터에 맞춰 만든 가상 정보입니다."
- 섹션: "## 기본 정보"(업종, 영업시간과 휴무일, 매장 전화를 받는 시간), "## 메뉴"(리뷰에 나온 메뉴를 중심으로 6~10개, 가격, 기본 제공 품목과 옵션), "## 배달·포장", "## 문제 대응"(주어진 policy를 따름), "## 리뷰 이벤트"(주어진 event를 따름).
- 리뷰에 자주 나오는 불만(배달 지연, 양, 맛, 포장 등)에 대해 가게가 안내할 수 있는 사실을 한두 줄 넣으세요.
- 600자~1200자 안팎으로 쓰세요.
출력은 {"document": "마크다운 문서"} 형식의 JSON 객체 하나뿐입니다."""


def check_document(value):
    doc = value.get("document")
    if set(value) != {"document"} or not isinstance(doc, str):
        raise ValueError('Expected {"document": "..."}')
    doc = doc.strip()
    if not 300 <= len(doc) <= 3000 or not doc.startswith("#") or "가상" not in doc:
        raise ValueError("Document must be 300-3000 characters, start with '#', and say it is fictional")
    return doc


def plan(rows, seed=42):
    """[(app_id, reviews, policy, event)] in app_id order; policy/event fixed per store by seed."""
    by_store = {}
    for row in rows:
        by_store.setdefault(row.app_id, []).append(row)
    rng = random.Random(seed)
    jobs = []
    for app_id in sorted(by_store):
        reviews = [r.review for r in by_store[app_id][:MAX_REVIEWS]]
        jobs.append((app_id, reviews, rng.choice(POLICY_STYLES), rng.choice(EVENT_STYLES)))
    return jobs


def generate(llm, jobs, out_dir, max_tokens=4096, attempts=3, overwrite=False):
    """Write <out_dir>/<app_id>/info.md per job. Existing documents are kept unless overwrite:
    pairs already built from a store were made with its current document."""
    out_dir = Path(out_dir)
    written = []
    for app_id, reviews, policy, event in jobs:
        path = out_dir / app_id / "info.md"
        if path.exists() and not overwrite:
            continue
        payload = {"reviews": reviews, "policy": policy, "event": event}
        doc = structured_call(
            llm,
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            check_document,
            max_tokens,
            attempts,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(doc + "\n", encoding="utf-8", newline="\n")
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description="Fictional store documents for public-review restaurants")
    parser.add_argument("--reviews", nargs="+", required=True, help="Review JSONL file(s)")
    parser.add_argument("--out", required=True, help="Output folder (one subfolder per store)")
    parser.add_argument("--backend", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--cache", default="outputs/synth_store_docs_cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Rewrite stores that already have info.md")
    args = parser.parse_args()
    rows = [r for path in args.reviews for r in load_custom_reviews(path)]
    jobs = plan(rows, args.seed)
    llm = CachedLLM(make_backend(args.backend, args.model, Config()), args.cache)
    print(f"{len(jobs)} stores, requests to {args.backend}:{args.model}", flush=True)
    written = generate(llm, jobs, args.out, overwrite=args.overwrite)
    kept = len(jobs) - len(written)
    print(f"Wrote {len(written)} store documents to {args.out}, kept {kept} existing (API calls {llm.calls})")


if __name__ == "__main__":
    main()
