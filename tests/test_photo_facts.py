import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest

from scrable.config import Config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("photo_facts", ROOT / "scripts" / "photo_facts.py")
photo_facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(photo_facts)


def test_build_refuses_unchecked_facts_and_maps_reviews_to_photos(tmp_path):
    facts = {
        "a.jpg": {"checked": True, "facts": "김밥 한 줄이 보입니다."},
        "b.jpg": {"checked": False, "facts": "라면 한 그릇이 보입니다."},
    }
    photo_of = {"r2": "b.jpg", "r1": "a.jpg", "r3": "a.jpg"}
    with pytest.raises(ValueError, match="not checked"):
        photo_facts.build(facts, photo_of)
    facts["b.jpg"]["checked"] = True
    result = photo_facts.build(facts, photo_of)
    assert result["predictions"] == [
        {"review_id": "r1", "photo_description": "김밥 한 줄이 보입니다."},
        {"review_id": "r2", "photo_description": "라면 한 그릇이 보입니다."},
        {"review_id": "r3", "photo_description": "김밥 한 줄이 보입니다."},
    ]
    with pytest.raises(ValueError, match="No facts"):
        photo_facts.build(facts, {"r4": "c.jpg"})


def test_repo_facts_cover_every_test_photo_and_eval_config_loads():
    facts = photo_facts.load_facts(ROOT / "configs" / "photo_facts.json")
    labels = json.loads((ROOT / "configs" / "photo_labels.json").read_text(encoding="utf-8"))
    assert set(facts) == {k for k in labels if not k.startswith("_")}
    assert all(entry["facts"].strip() for entry in facts.values())
    config = Config.load(ROOT / "configs" / "photo_eval_facts.json")
    assert config.photo_descriptions_from == "outputs/photo_facts/descriptions.json"
    assert config.image_model is None  # fixed facts only: no vision model is loaded
    haiku = Config.load(ROOT / "configs" / "photo_eval_haiku_facts.json")
    assert haiku.photo_descriptions_from == config.photo_descriptions_from


def test_sheet_and_tally_count_use_of_photo_information(tmp_path):
    def run(text, description, score):
        return {
            "r1": {"review_id": "r1", "review": "맛있어요", "image": "data/x/a.jpg",
                   "photo_description": description, "prediction": text, "score": score},
        }

    columns, rows = photo_facts.sheet_rows(run("감사합니다", "음식", 0.6), run("김밥 감사", "김밥", 0.7))
    assert rows[0]["사진"] == "a.jpg" and rows[0]["③ 사람 확인 사실"] == "김밥"
    assert columns[-len(photo_facts.FILL_IN):] == photo_facts.FILL_IN

    filled = [
        {photo_facts.ADDS: "예", photo_facts.VISION_HAS: "아니오", photo_facts.VISION_USED: "아니오",
         photo_facts.FACTS_USED: "예", photo_facts.VISION_UNSUPPORTED: "아니오",
         photo_facts.FACTS_UNSUPPORTED: "아니오"},
        {photo_facts.ADDS: "아니오", photo_facts.VISION_UNSUPPORTED: "예", photo_facts.FACTS_UNSUPPORTED: ""},
    ]
    result = photo_facts.tally(filled)
    assert result["photo_adds_info"] == (1, 2, 0)
    assert result["vision_reply_uses_it"] == (0, 1, 0)
    assert result["facts_reply_uses_it"] == (1, 1, 0)
    assert result["vision_reply_unsupported"] == (1, 2, 0)
    assert result["facts_reply_unsupported"] == (0, 1, 1)


def test_read_sheet_accepts_excel_korean_csv(tmp_path):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["review_id", photo_facts.ADDS])
    writer.writeheader()
    writer.writerow({"review_id": "r1", photo_facts.ADDS: "예"})
    path = tmp_path / "sheet.csv"
    path.write_bytes(buffer.getvalue().encode("cp949"))
    assert photo_facts.read_sheet(path)[0][photo_facts.ADDS] == "예"
