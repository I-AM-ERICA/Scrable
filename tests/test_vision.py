"""Vision-3B loader helpers (no model download) and the per-reply speed report."""

import numpy as np
import pytest

from scrable import Config
from scrable.evaluate import seconds_summary
from scrable.llm import clean_completion, is_vision, make_backend, new_tokens, vision_messages

VISION = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"


def test_vision_model_is_detected_and_messages_become_typed_content():
    assert is_vision(VISION) and not is_vision("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B")
    assert not is_vision("some-org/other-vision-model")
    chat = vision_messages([{"role": "system", "content": "규칙"}, {"role": "user", "content": "리뷰"}])
    assert chat[1] == {"role": "user", "content": [{"type": "text", "text": "리뷰"}]}


def test_generated_tokens_and_end_markers_are_stripped():
    prompt = np.array([[1, 2, 3]])
    assert new_tokens(np.array([1, 2, 3, 7, 8]), prompt).tolist() == [7, 8]  # Prompt echoed back.
    assert new_tokens(np.array([7, 8]), prompt).tolist() == [7, 8]  # Only new tokens returned.
    assert clean_completion("감사합니다.<|im_end|><|endofturn|>잡음") == "감사합니다."


def test_vision_model_refuses_a_lora_adapter_before_loading():
    with pytest.raises(ValueError, match="not supported yet"):
        make_backend("local", VISION, Config(knowledge_size=0, documents_dir="d"), "outputs/stage2/adapter")


def test_seconds_summary():
    speed = seconds_summary([0.5, 1.0, 4.0, 2.0])
    assert speed == {"mean": 1.875, "p90": 4.0, "max": 4.0}
