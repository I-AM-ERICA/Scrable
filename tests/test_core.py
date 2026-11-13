import json
from dataclasses import replace

import pytest

from scrable import Config
from scrable.data import parse_aarsynth, split_reviews
from scrable.llm import CachedLLM, structured_call
from scrable.pipeline import identify_ir, validate_judgment
from scrable.types import CRITERIA, Feedback, weighted_score
from tests.helpers import reviews


def feedback(n):
    return [
        Feedback(str(i), "review", "app", "prediction", {k: 1 + i % 5 for k in CRITERIA}, [], {})
        for i in range(n)
    ]


def test_weighted_score_accuracy_double_and_endpoints():
    assert weighted_score({k: 1 for k in CRITERIA}) == 0
    assert weighted_score({k: 5 for k in CRITERIA}) == 1
    assert weighted_score({"relevancy": 1, "specificity": 1, "accuracy": 5, "grammar": 1}) == 0.4


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, 6, True, "5"])
def test_scores_reject_invalid_values(value):
    with pytest.raises(ValueError):
        weighted_score({k: value for k in CRITERIA})


def test_ir_lowest_and_random_are_disjoint_and_reproducible():
    rows = feedback(100)
    picked = identify_ir(rows)
    assert len(picked) == len({f.review_id for f in picked}) == 40
    assert picked[:30] == sorted(rows, key=lambda f: (f.score, f.review_id))[:30]
    assert picked == identify_ir(rows)
    assert picked[30:] != identify_ir(rows, seed=99)[30:]
    assert len(identify_ir(rows[:1])) == 1
    assert len(identify_ir(rows[:3])) == 2


def test_split_removes_duplicates_before_partitioning():
    rows = reviews()
    rows += [replace(rows[0], id="duplicate", review="  FOOD  delivery review NUMBER 0 was delayed ")]
    config = Config(train_size=4, validation_size=3, test_size=3, knowledge_size=10)
    split = split_reviews(rows, config)
    assert split == split_reviews(rows, config)
    ids = [r.id for group in split.values() for r in group]
    texts = [r.review.casefold().strip() for group in split.values() for r in group]
    assert len(ids) == len(set(ids)) == 20
    assert len(texts) == len(set(texts))
    with pytest.raises(ValueError):
        split_reviews(rows[:3], config)


def test_real_aarsynth_format_and_invalid_lines():
    row = parse_aarsynth(
        "com.ubercab.eats***2***2019-10-26***None***most restaurants are greyed out***"
        " <salutation> please share more details about your concern***2019-10-28"
    )
    assert row.app_id == "com.ubercab.eats"
    assert row.review == "most restaurants are greyed out"
    assert "salutation" not in row.expert_response
    assert parse_aarsynth("broken data") is None
    assert parse_aarsynth("a***9***date***None***a review with words***a response with words***date") is None


def test_structured_output_retries_without_fabricating_score():
    class BrokenThenValid:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, max_tokens):
            self.calls += 1
            return (
                "not json"
                if self.calls == 1
                else json.dumps({"score": 4, "reason": "Supported", "suggestions": []})
            )

    llm = BrokenThenValid()
    result = structured_call(llm, [{"role": "user", "content": "test"}], validate_judgment, 100)
    assert result["score"] == 4 and llm.calls == 2
    with pytest.raises(ValueError, match="after 1 attempts"):
        structured_call(BrokenThenValid(), [], validate_judgment, 100, attempts=1)


def test_cache_keys_include_prompt_model_and_budget(tmp_path):
    class Backend:
        identity = {"model": "a"}
        calls = 0

        def complete(self, messages, max_tokens):
            self.calls += 1
            return "reply"

    backend = Backend()
    llm = CachedLLM(backend, tmp_path)
    llm.complete([], 10)
    llm.complete([], 10)
    llm.complete([], 11)
    llm.complete([{"role": "user", "content": "changed"}], 10)
    backend.identity = {"model": "b"}
    llm.complete([], 10)
    assert backend.calls == 4
    assert llm.hits == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 5},
        {"train_size": 0},
        {"max_iterations": -1},
        {"lowest_fraction": 0.9, "random_fraction": 0.2},
        {"chunk_overlap": 500},
    ],
)
def test_invalid_config_fails_early(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)
