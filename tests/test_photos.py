"""Photo reviews (plan A): photo field, descriptions in the generator input, photo for the judge."""

import base64
import json

import pytest

from scrable import Config
from scrable.data import load_custom_reviews
from scrable.evaluate import evaluate
from scrable.photos import PhotoDescriber, image_block
from scrable.pipeline import generation_messages
from scrable.prompts import PHOTO_RULES
from scrable.rag import HybridRetriever
from scrable.synth_photo_reviews import find_photos, generate
from scrable.types import Document, Review
from tests.helpers import TestEmbeddings

FAKE_PNG = b"\x89PNG\r\n\x1a\nfake-image-bytes"


class FakeVision:
    identity = {"backend": "test-only-vision"}

    def __init__(self):
        self.calls = []

    def describe_image(self, path, question, max_tokens):
        self.calls.append(str(path))
        return "떡볶이 위에 머리카락으로 보이는 이물질이 있습니다."


class PhotoJudge:
    """Test-only API judge that records whether it saw an image block."""

    identity = {"backend": "test-only-photo-judge"}

    def __init__(self):
        self.contents = []

    def complete(self, messages, max_tokens):
        content = messages[-1]["content"]
        self.contents.append(content)
        if isinstance(content, list):
            return json.dumps({"score": 4, "reason": "fixture", "suggestions": []})
        if "criterion" in content:
            return json.dumps({"score": 2, "reason": "fixture", "suggestions": []})
        return "답글입니다"


def photo(tmp_path, name="a.png"):
    path = tmp_path / name
    path.write_bytes(FAKE_PNG)
    return path


def test_review_photo_field_is_optional_and_validated(tmp_path):
    data = tmp_path / "r.jsonl"
    data.write_text(
        json.dumps(
            {"id": "p1", "review": "이물질이 나왔어요", "app_id": "s", "image": "x.png"}, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    assert load_custom_reviews(data)[0].image == "x.png"
    assert Review("r", "맛있어요", "s").image is None
    with pytest.raises(ValueError, match="image"):
        Review("r", "맛있어요", "s", image=" ")
    with pytest.raises(ValueError, match="image_model"):
        Config(knowledge_size=0, documents_dir="d", image_model="some/other-model")


def test_photo_description_changes_generator_input_only_when_present():
    review = Review("r", "이물질이 나왔어요", "s", rating=1, image="x.png")
    plain = generation_messages("p", review, [])
    assert plain == generation_messages("p", Review("r", "이물질이 나왔어요", "s", rating=1), [])
    with_photo = generation_messages("p", review, [], photo_description="머리카락이 보입니다")
    assert json.loads(with_photo[1]["content"])["photo_description"] == "머리카락이 보입니다"
    assert with_photo[0]["content"].endswith(PHOTO_RULES)


def test_photo_descriptions_are_cached_by_image_content(tmp_path):
    vision = FakeVision()
    describer = PhotoDescriber(vision, tmp_path / "cache")
    first, second = photo(tmp_path, "a.png"), photo(tmp_path, "b.png")  # Same bytes.
    assert describer.describe(first) == describer.describe(second)
    assert len(vision.calls) == 1 and describer.calls == 1
    block = image_block(first)
    assert block["source"]["media_type"] == "image/png"
    assert base64.b64decode(block["source"]["data"]) == FAKE_PNG
    with pytest.raises(ValueError, match="Unsupported image type"):
        image_block(tmp_path / "a.gif")


def test_photo_review_evaluation_describes_photo_and_shows_it_to_judge(tmp_path):
    image = photo(tmp_path)
    data = tmp_path / "photo_reviews.jsonl"
    data.write_text(
        json.dumps(
            {
                "id": "photo-app-a",
                "app_id": "app",
                "rating": 1,
                "review": "이물질이 나왔어요",
                "image": str(image),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    docs = tmp_path / "docs" / "app"
    docs.mkdir(parents=True)
    (docs / "info.md").write_text("환불은 확인 후 안내", encoding="utf-8")
    config = Config(
        judge_backend="anthropic",
        judge_model="judge",
        knowledge_size=0,
        documents_dir=str(tmp_path / "docs"),
        custom_data=str(data),
        image_model="naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
        output_dir=str(tmp_path / "out"),
    )
    rag = HybridRetriever(
        [Document("d", "환불은 확인 후 안내", "app", "doc")], TestEmbeddings(), tmp_path / "rag"
    )
    judge = PhotoJudge()
    describer = PhotoDescriber(FakeVision(), tmp_path / "cache")
    metrics = evaluate(
        config, "with-photos", None, None, (judge, judge), rag, split="all", describer=describer
    )
    assert metrics["summary"]["count"] == 1 and metrics["image_model"] == config.image_model
    generator_input = json.loads(judge.contents[0])
    assert generator_input["photo_description"].startswith("떡볶이")
    judged = [c for c in judge.contents if isinstance(c, list)]
    assert len(judged) == 4 and all(c[0]["type"] == "image" for c in judged)
    saved = json.loads((tmp_path / "out" / "all_predictions.json").read_text(encoding="utf-8"))
    assert saved["predictions"][0]["photo_description"].startswith("떡볶이")

    local_judge = Config(**{**config.to_dict(), "judge_backend": None, "judge_model": None})
    with pytest.raises(ValueError, match="judge_backend=anthropic"):
        evaluate(local_judge, "x", None, None, (judge, judge), rag, split="all", describer=describer)


class ReviewWriter:
    identity = {"backend": "test-only-photo-review-writer"}

    def complete(self, messages, max_tokens):
        assert messages[-1]["content"][0]["type"] == "image"
        return json.dumps({"rating": 2, "review": "포장이 다 흐트러져서 왔어요"}, ensure_ascii=False)


def test_synthetic_photo_reviews_one_per_photo(tmp_path):
    for store, name in (("bunsik-01", "a.png"), ("bunsik-01", "b.jpg"), ("cafe-01", "c.png")):
        (tmp_path / "photos" / store).mkdir(parents=True, exist_ok=True)
        (tmp_path / "photos" / store / name).write_bytes(FAKE_PNG)
    (tmp_path / "photos" / "bunsik-01" / "notes.txt").write_text("not a photo", encoding="utf-8")
    photos = find_photos(tmp_path / "photos")
    assert [(s, p.name) for s, p in photos] == [
        ("bunsik-01", "a.png"),
        ("bunsik-01", "b.jpg"),
        ("cafe-01", "c.png"),
    ]
    docs = [Document("d1", "분식", "bunsik-01", "info.md"), Document("d2", "카페", "cafe-01", "info.md")]
    rows = generate(ReviewWriter(), docs, photos)
    assert [r["id"] for r in rows] == ["photo-bunsik-01-a", "photo-bunsik-01-b", "photo-cafe-01-c"]
    assert all(Review(**r).image.endswith((".png", ".jpg")) for r in rows)
    with pytest.raises(ValueError, match="No store documents"):
        generate(ReviewWriter(), docs[:1], photos)


class StyleWriter:
    """Test-only writer: answers each style's request, once with a rating the style forbids."""

    identity = {"backend": "test-only-style-writer"}

    def __init__(self):
        self.systems = []

    def complete(self, messages, max_tokens):
        system = messages[0]["content"]
        self.systems.append(system)
        retry = len(messages) > 2  # structured_call appended the validation error.
        if "별점은 4나 5" in system:
            rating, text = (5, "잘 먹었어요") if retry else (2, "잘 먹었어요")
        elif "별점은 1~3" in system:
            rating, text = 2, "음식이 다 식어서 왔어요"
        else:
            rating, text = 5, "튀김이 바삭해 보여요"
        return json.dumps({"rating": rating, "review": text}, ensure_ascii=False)


def test_photo_review_styles_add_rows_with_style_ratings(tmp_path):
    from scrable.synth_photo_reviews import SYSTEM, style_of

    (tmp_path / "photos" / "bunsik-01").mkdir(parents=True)
    (tmp_path / "photos" / "bunsik-01" / "a.png").write_bytes(FAKE_PNG)
    docs = [Document("d1", "분식", "bunsik-01", "info.md")]
    writer = StyleWriter()
    rows = generate(writer, docs, find_photos(tmp_path / "photos"), styles=("photo", "short", "complaint"))
    assert [r["id"] for r in rows] == [
        "photo-bunsik-01-a",
        "photo-bunsik-01-a-short",
        "photo-bunsik-01-a-complaint",
    ]
    assert [r["rating"] for r in rows] == [5, 5, 2]  # The short review's first rating (2) was rejected.
    assert writer.systems[0] == SYSTEM  # The first test's requests (and cache) are unchanged.
    assert [style_of(r["id"]) for r in rows] == ["photo", "short", "complaint"]
    with pytest.raises(ValueError, match="Unknown style"):
        generate(writer, docs, find_photos(tmp_path / "photos"), styles=("long",))
