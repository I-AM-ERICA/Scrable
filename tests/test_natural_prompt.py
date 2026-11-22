from pathlib import Path

from scrable.config import Config
from scrable.experiment import load_base_prompt

ROOT = Path(__file__).resolve().parents[1]


def test_natural_prompt_keeps_every_safety_rule_of_the_stage4_prompt():
    old = (ROOT / "prompts" / "stage4_prompt.txt").read_text(encoding="utf-8").splitlines()
    new = (ROOT / "prompts" / "stage4_natural_prompt.txt").read_text(encoding="utf-8").splitlines()
    changed = {line for line in old if line not in new}
    # Only the two lines behind echoed reviews and the fixed "확인 후 개별 안내드리겠습니다" phrase change.
    assert len(changed) == 2 and all("짚어" in line or "개별 안내" in line for line in changed)
    for rule in ("리뷰 작성을 조건으로 보상을 제시하지 마세요", "가게 정보에 근거가 있을 때만", "원인이나 책임을 단정하지"):
        assert any(rule in line for line in new)


def test_natural_photo_eval_uses_the_same_descriptions_and_no_vision_model():
    config = Config.load(ROOT / "configs" / "photo_eval_natural.json")
    assert config.photo_descriptions_from == "outputs/photo_eval_stage4/all_predictions.json"
    assert config.image_model is None and config.response_adapter == "outputs/stage4_photo/adapter"
    assert "사장님이 손님에게 말하듯" in load_base_prompt(config)
