"""Stage 4: public review CSV conversion, fictional store documents, complaint-only synthetic counts."""

import csv
import json

import pytest

from scrable.data import load_custom_reviews, load_documents
from scrable.public_reviews import build, keep, read_csv
from scrable.synth_reviews import parse_counts, plan
from scrable.synth_store_docs import generate
from scrable.synth_store_docs import plan as doc_plan
from scrable.types import Review


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


# ---------------------------------------------------------------- photo reviews

FAKE_PNG = b"\x89PNG\r\n\x1a\nfake-image-bytes"


def csv_row(store, menu, text, photo="1", star="5"):
    return {"RestaurantID": store, "Menu": menu, "Review": text, "Total": star, "HasPicture": photo}


def test_menu_matching_uses_dish_names_only():
    from scrable.public_photo_reviews import food_of

    foods = ["김치찌개", "라면", "냉면", "물냉면", "부대찌개"]
    assert food_of("[2~3인분] 짜글이 정식/1(짜글이 선택（생돼지 김치찌개 짜글이）)", foods) is None
    assert food_of("부대찌개/1(라면사리 추가)", foods) == "부대찌개"
    assert food_of("짜글이/1,라면사리/1", foods) is None
    assert food_of("물냉면/2", foods) == "물냉면"
    # Only the main dish counts: event items are skipped, side orders and longer dishes ignored.
    more = ["식혜", "육회", "순대", "새우튀김", "찜닭"]
    assert food_of("제육볶음세트/1,공기밥/1,식혜238mL/1", more) is None
    assert food_of("스시고칸100원의행복[찜&후기작성]2VENT/1,육회비빔밥/1", more) is None
    assert food_of("순대국/1,설렁탕/1", more) is None
    assert food_of("[프리미엄]까만／빨간찜닭/1,15cm왕새우튀김2개/2", more) == "찜닭"


def test_reviews_about_another_dish_are_skipped():
    from scrable.public_photo_reviews import names_other_food

    foods = ["계란찜", "해물찜", "순대"]
    assert names_other_food("해물찜 시켰는데 너무 맵네요", "계란찜", foods)
    assert not names_other_food("계란찜 부드러워요", "계란찜", foods)
    assert not names_other_food("순대국 정식의 순대 맛있어요", "순대", foods)


def test_photo_reviews_get_dish_photos_and_skip_test_stores(tmp_path):
    from scrable.public_photo_reviews import build, find_food_photos

    for food in ("떡볶이", "족발"):
        (tmp_path / "photos" / "분류" / food).mkdir(parents=True)
        for i in range(2):
            (tmp_path / "photos" / "분류" / food / f"{i}.jpg").write_bytes(FAKE_PNG + bytes([i]))
    rows = [csv_row("s1", "떡볶이/1", f"떡볶이 너무 맛있어요 사진 첨부합니다 {i}") for i in range(4)]
    rows += [csv_row("s2", "족발/1(소)", f"족발 부드럽고 양도 많아요 {i}", star="3") for i in range(4)]
    rows += [csv_row("s3", "족발/1", "테스트 식당 족발 리뷰입니다 맛있어요")]
    rows += [csv_row("s1", "떡볶이/1", "사진 없는 떡볶이 리뷰입니다 맛있어요", photo="0")]
    store_map = {"s1": "pub-001", "s3": "pubtest-001"}
    out, new_ids, stats = build(rows, find_food_photos(tmp_path / "photos"), store_map, count=6, per_food=3)
    assert len(out) == 6 and stats["by_food"] == {"떡볶이": 3, "족발": 3}
    assert {r["app_id"] for r in out} == {"pub-001", "pubimg-001"} and new_ids == {"s2": "pubimg-001"}
    assert all("/떡볶이/" in r["image"] for r in out if r["app_id"] == "pub-001")
    assert all(Review(**r).image for r in out) and len({r["id"] for r in out}) == 6
    assert "테스트 식당" not in json.dumps(out, ensure_ascii=False)


def test_existing_store_documents_are_kept(tmp_path):
    jobs = [("pub-001", ["리뷰"], "policy", "event"), ("pubimg-001", ["리뷰"], "policy", "event")]
    (tmp_path / "pub-001").mkdir()
    (tmp_path / "pub-001" / "info.md").write_text("기존 문서", encoding="utf-8")
    written = generate(DocLLM(), jobs, tmp_path)
    assert written == [tmp_path / "pubimg-001" / "info.md"]
    assert (tmp_path / "pub-001" / "info.md").read_text(encoding="utf-8") == "기존 문서"


class PhotoPairJudge:
    """Test-only judge: sees the photo only for relevancy/accuracy; teacher answers score higher."""

    identity = {"backend": "test-only-photo-pair-judge"}

    def __init__(self):
        self.with_photo = []

    def complete(self, messages, max_tokens):
        content = messages[-1]["content"]
        if isinstance(content, list):
            payload = json.loads(content[1]["text"].split("\n", 1)[1])
            self.with_photo.append(payload["criterion"])
        else:
            payload = json.loads(content)
        score = 5 if "teacher" in payload["prediction"] else 2
        return json.dumps({"score": score, "reason": "fixture", "suggestions": []})


class PhotoTeacher:
    identity = {"backend": "test-only-teacher"}

    def complete(self, messages, max_tokens):
        return "teacher answer about the dish"


class Sampler:
    identity = {"backend": "test-only-sampler"}

    def sample(self, messages, max_tokens, n, temperature, top_p, seed):
        return [f"small answer {i}" for i in range(n)]


class FakeDescriber:
    def __init__(self):
        self.paths = []

    def describe(self, path, food=None):
        self.paths.append((path, food))
        return f"빨간 {food or '떡볶이'} 한 접시가 보입니다."


def test_photo_pairs_put_description_in_generator_input(tmp_path):
    from scrable.dpo import DPOConfig, build_pairs
    from scrable.rag import HybridRetriever
    from scrable.types import Document
    from tests.helpers import TestEmbeddings

    image = tmp_path / "a.png"
    image.write_bytes(FAKE_PNG)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("프롬프트", encoding="utf-8")
    dcfg = DPOConfig(
        base={
            "judge_backend": "anthropic",
            "judge_model": "judge",
            "custom_data": "unused.jsonl",
            "documents_dir": "docs",
            "knowledge_size": 0,
            "base_prompt_file": str(prompt),
            "output_dir": str(tmp_path / "out"),
            "image_model": "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
        },
        teacher_models=[{"backend": "anthropic", "model": "big"}],
    )
    rows = [
        Review("p1", "떡볶이 맛있어요", "store", rating=5, image=str(image)),
        Review("t1", "배달이 빨랐어요", "store", rating=5),
    ]
    rag = HybridRetriever([Document("d", "메뉴 안내", "store", "doc")], TestEmbeddings(), tmp_path / "rag")
    judge, describer = PhotoPairJudge(), FakeDescriber()
    summary = build_pairs(dcfg, rows, rag, Sampler(), judge, (PhotoTeacher(),), describer)
    assert summary["pairs"] == 2 and describer.paths == [(str(image), None)]
    pairs = {
        p["review_id"]: p for p in map(json.loads, dcfg.pairs_file.read_text(encoding="utf-8").splitlines())
    }
    photo_input = json.loads(pairs["p1"]["messages"][1]["content"])
    assert photo_input["photo_description"].startswith("빨간 떡볶이")
    assert "photo_description" not in json.loads(pairs["t1"]["messages"][1]["content"])
    # 4 candidates (3 samples + teacher) x 2 photo criteria for the one photo review.
    assert sorted(set(judge.with_photo)) == ["accuracy", "relevancy"] and len(judge.with_photo) == 8


def test_folder_name_is_given_to_describer(tmp_path):
    from scrable.photos import HINT_PROMPT, PhotoDescriber

    class Vision:
        identity = {"backend": "test-only-vision"}

        def __init__(self):
            self.prompts = []

        def describe_image(self, path, prompt, max_tokens):
            self.prompts.append(prompt)
            return "설명"

    photo = tmp_path / "갈치구이" / "a.png"
    photo.parent.mkdir()
    photo.write_bytes(FAKE_PNG)
    vision = Vision()
    describer = PhotoDescriber(vision, tmp_path / "cache")
    describer.describe(photo, "갈치구이")
    describer.describe(photo)  # Different prompt -> separate cache entry.
    assert vision.prompts[0] == HINT_PROMPT.format(food="갈치구이") and len(vision.prompts) == 2
