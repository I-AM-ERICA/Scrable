import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("best_of_n", ROOT / "scripts" / "best_of_n.py")
best_of_n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(best_of_n)


def candidate(source, text, score, spec):
    return {"source": source, "text": text, "score": score,
            "scores": {"relevancy": 4, "specificity": spec, "accuracy": 4, "grammar": 4}}


def row(review, docs, candidates):
    payload = {"app_id": "s", "review": review, "retrieved_documents": [{"id": "d", "source": "x", "text": docs}]}
    return {"review": review, "messages": [{"role": "system", "content": "s"},
                                           {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            "candidates": candidates}


def test_pickers_and_gains():
    rows = [
        row("빵 맛있어요", "소금빵 크루아상 주차 가능", [
            candidate("policy", "감사합니다", 0.4, 2),
            candidate("policy", "소금빵 좋아해 주셔서 감사합니다. 주차 가능", 0.8, 4),
            candidate("policy", "정말 정말 진심으로 감사드립니다 다음에도 또 꼭 오세요 고객님", 0.5, 2),
            candidate("teacher:anthropic:claude-haiku-5-5", "소금빵 감사", 0.9, 5),
        ]),
        row("좋아요", "국밥", [candidate("policy", "감사", 0.5, 3)]),  # one reply: skipped
    ]
    result = best_of_n.summarize(rows)
    assert result["reviews"] == 1 and result["skipped"] == 1
    random_overall, random_spec, _ = result["random"]
    assert abs(random_overall - (0.4 + 0.8 + 0.5) / 3) < 1e-9
    assert abs(random_spec - (0.25 + 0.75 + 0.25) / 3) < 1e-9
    assert result["best"][0] == 0.8
    assert result["doc words"][0] == 0.8  # uses 주차 and 가능 from the store document
    assert result["longest"][0] == 0.5
    assert result["haiku"][0] == 0.9
    gain, low, high = result["best"][2]
    assert abs(gain - (0.8 - 1.7 / 3)) < 1e-9 and low == high == gain
