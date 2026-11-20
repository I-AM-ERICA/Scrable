"""Stage 4: public review CSV conversion, fictional store documents, complaint-only synthetic counts."""

import csv
import json

import pytest

from scrable.data import load_custom_reviews, load_documents
from scrable.public_reviews import build, keep, read_csv
from scrable.synth_reviews import parse_counts, plan
from scrable.synth_store_docs import generate
from scrable.synth_store_docs import plan as doc_plan


def fake_rows(stores=6, per_store=12):
    rows = []
    for s in range(stores):
        for i in range(per_store):
            star = 5 if i % 3 else 2
            text = f"가게{s} 떡볶이가 정말 맛있었고 배달도 빨랐어요 {i}번째 주문입니다"
            rows.append({"Restaurant": f"real-name-{s}", "Review": text, "Star": str(star)})
    return rows


def test_keep_filters_personal_data_and_non_korean():
    assert keep("떡볶이가 맛있어요 또 시킬게요")
    assert not keep("짧음")
    assert not keep("연락 주세요 010-1234-5678 맛있어요")
    assert not keep("주문번호 확인 부탁드려요 맛은 좋아요")
    assert not keep("very good food delivery fast 좋아")
    assert not keep("사이트 www.example.com 에서 봤어요 맛있어요")


def test_build_splits_by_restaurant_and_hides_names():
    rows = fake_rows()
    rows.append(dict(rows[0]))  # exact duplicate
    train, test, ids, stats = build(rows, "Review", "Restaurant", "Star", 3, 6, 2, 4)
    assert len(train) == 18 and len(test) == 8
    assert stats["dropped"]["duplicate"] == 1
    assert not {r["app_id"] for r in train} & {r["app_id"] for r in test}
    assert all(r["app_id"].startswith("pub-") for r in train)
    assert all(r["app_id"].startswith("pubtest-") for r in test)
    assert "real-name" not in json.dumps(train + test, ensure_ascii=False)
    assert len({r["id"] for r in train + test}) == len(train + test)
    # Non-5-star reviews are taken first, up to half of each store's sample.
    for app_id in {r["app_id"] for r in train}:
        store = [r for r in train if r["app_id"] == app_id]
        assert sum(r["rating"] < 5 for r in store) == 3


def test_more_training_stores_keep_test_set_and_first_stores():
    rows = fake_rows(stores=8)
    train_small, test_small, _, _ = build(rows, "Review", "Restaurant", "Star", 3, 6, 2, 4)
    train_big, test_big, _, _ = build(rows, "Review", "Restaurant", "Star", 5, 6, 2, 4)
    assert test_small == test_big
    assert train_big[: len(train_small)] == train_small


def test_build_without_rating_and_too_few_stores():
    train, _, _, _ = build(fake_rows(), "Review", "Restaurant", None, 2, 5, 1, 3)
    assert all(r["rating"] is None for r in train)
    with pytest.raises(ValueError, match="restaurants"):
        build(fake_rows(stores=2), "Review", "Restaurant", None, 2, 5, 1, 3)
    with pytest.raises(ValueError, match="Column"):
        build(fake_rows(), "Text", "Restaurant")


def test_read_csv_and_jsonl_round_trip(tmp_path):
    path = tmp_path / "reviews.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Restaurant", "Review", "Star"])
        writer.writeheader()
        writer.writerows(fake_rows())
    rows = read_csv(path)
    assert list(rows[0]) == ["Restaurant", "Review", "Star"]
    train, _, _, _ = build(rows, "Review", "Restaurant", "Star", 2, 4, 1, 2)
    out = tmp_path / "train.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8")
    assert len(load_custom_reviews(out)) == 8


class DocLLM:
    identity = {"backend": "test-only"}

    def __init__(self):
        self.payloads = []

    def complete(self, messages, max_tokens):
        payload = json.loads(messages[-1]["content"])
        self.payloads.append(payload)
        body = "가상 메뉴와 정책입니다. " * 30
        return json.dumps({"document": f"# 새이름분식 가게 정보\n※ 실험용 가상 매장입니다.\n{body}"})


def test_store_docs_one_folder_per_store(tmp_path):
    reviews = tmp_path / "r.jsonl"
    train, _, _, _ = build(fake_rows(), "Review", "Restaurant", "Star", 3, 4, 1, 2)
    reviews.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8")
    jobs = doc_plan(load_custom_reviews(reviews))
    llm = DocLLM()
    written = generate(llm, jobs, tmp_path / "docs")
    assert len(written) == 3
    assert all(len(p["reviews"]) == 4 and p["policy"] and p["event"] for p in llm.payloads)
    docs = load_documents(tmp_path / "docs", "default")
    assert {d.app_id for d in docs} == {r["app_id"] for r in train}


def test_complaint_only_counts():
    assert parse_counts("2:3,1:2") == {2: 3, 1: 2}
    with pytest.raises(ValueError):
        parse_counts("7:1")
    batches = plan(["pub-001", "pub-002"], parse_counts("2:3,1:2"))
    assert sum(len(specs) for _, _, specs in batches) == 10
    assert {rating for _, rating, _ in batches} == {1, 2}
