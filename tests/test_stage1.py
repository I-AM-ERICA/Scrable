"""Stage 1: strong API judge/prompt generator, small response generator, comparison models."""

import json
from types import SimpleNamespace

import pytest

import scrable.llm as llm_module
from scrable import Config
from scrable.anthropic_llm import FALLBACK_BETA, AnthropicLLM
from scrable.data import knowledge_documents, load_documents, split_reviews
from scrable.experiment import run_experiment, score_table
from scrable.pipeline import ScoredResponseGenerator
from scrable.rag import HybridRetriever
from scrable.types import Document, Review
from tests.helpers import ScriptedLLM, TestEmbeddings, reviews

API_JUDGE = {"judge_backend": "anthropic", "judge_model": "big", "prompt_backend": "anthropic"}


def test_roles_default_to_generator_and_validate():
    config = Config()
    assert config.role("response") == config.role("judge") == config.role("prompt")
    config = Config(**API_JUDGE, prompt_model="big")
    assert config.role("response") == ("local", config.model)
    assert config.role("judge") == config.role("prompt") == ("anthropic", "big")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"judge_backend": "anthropic"},  # API judge cannot silently reuse the local model ID.
        {"prompt_backend": "openai"},
        {"judge_backend": "gemini", "judge_model": "x"},
        {"judge_workers": 2},  # Local judge model is not thread-safe.
        {"knowledge_size": 0},  # No RAG source left.
        {"compare_models": [{"backend": "x", "model": "m"}]},
        {"compare_models": [{"backend": "anthropic"}]},
        {"anthropic_effort": "ultra"},
        {"api_min_output_tokens": 0},
    ],
)
def test_invalid_stage1_config_fails_early(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_knowledge_size_zero_allowed_with_documents():
    assert Config(knowledge_size=0, documents_dir="docs").knowledge_size == 0


def test_build_llms_routes_roles_and_shares_identical_specs(monkeypatch, tmp_path):
    built = []

    def fake_backend(backend, model, config):
        built.append((backend, model))
        return SimpleNamespace(identity={"backend": backend, "model": model})

    monkeypatch.setattr(llm_module, "make_backend", fake_backend)
    config = Config(**API_JUDGE, prompt_model="big", output_dir=str(tmp_path))
    shared = {}
    response, judge, prompt = llm_module.build_llms(config, shared=shared)
    assert response.backend.identity == {"backend": "local", "model": config.model}
    assert judge is prompt and judge is not response
    (compare,) = llm_module.build_llms(config, [("anthropic", "big")], shared)
    assert compare is judge
    assert built == [("local", config.model), ("anthropic", "big")]


class FakeAnthropic:
    def __init__(self, stop_reason="end_turn"):
        self.calls = []
        self.stop_reason = stop_reason
        self.messages = SimpleNamespace(create=lambda **kw: self._create("stable", kw))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: self._create("beta", kw)))

    def _create(self, endpoint, kwargs):
        self.calls.append((endpoint, kwargs))
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            stop_details=SimpleNamespace(explanation="test"),
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=" hi "),
            ],
        )


def test_anthropic_backend_request_shape():
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "review"}]
    client = FakeAnthropic()
    assert AnthropicLLM("claude-opus-5-5", 16000, client=client).complete(messages, 256) == "hi"
    endpoint, kwargs = client.calls[0]
    assert endpoint == "beta" and kwargs["betas"] == [FALLBACK_BETA] and kwargs["fallbacks"] == "default"
    assert kwargs["max_tokens"] == 16000 and kwargs["system"] == "rules"
    assert kwargs["messages"] == [{"role": "user", "content": "review"}]
    assert "output_config" not in kwargs and "thinking" not in kwargs and "temperature" not in kwargs

    client = FakeAnthropic()
    AnthropicLLM("claude-haiku-5-5", 1000, "low", client=client).complete(messages[1:], 2000)
    endpoint, kwargs = client.calls[0]
    assert endpoint == "stable" and "fallbacks" not in kwargs and "system" not in kwargs
    assert kwargs["max_tokens"] == 2000 and kwargs["output_config"] == {"effort": "low"}


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_anthropic_backend_never_returns_partial_output(stop_reason):
    llm = AnthropicLLM("claude-opus-5-5", client=FakeAnthropic(stop_reason))
    with pytest.raises(RuntimeError):
        llm.complete([{"role": "user", "content": "review"}], 256)


def test_parallel_judging_matches_sequential(tmp_path):
    rag = HybridRetriever([Document("d", "delivery facts", "app", "doc")], TestEmbeddings(), tmp_path)
    rows = reviews(5)
    results = []
    for workers in (1, 4):
        config = Config(judge_backend="openai", judge_model="big", judge_workers=workers)
        feedback = ScoredResponseGenerator(ScriptedLLM(), ScriptedLLM(), rag, config)("base", rows)
        results.append([f.to_dict() for f in feedback])
    assert results[0] == results[1]
    assert [f["review_id"] for f in results[0]] == [r.id for r in rows]


def test_rating_reaches_generator_and_judge(tmp_path):
    llm = ScriptedLLM()
    rag = HybridRetriever([Document("d", "delivery facts", "store", "doc")], TestEmbeddings(), tmp_path)
    row = Review("r1", "배달이 늦었어요", "store", None, rating=2)
    ScoredResponseGenerator(llm, llm, rag, Config())("base", [row])
    payloads = [json.loads(m[-1]["content"]) for m in llm.messages]
    assert payloads and all(p["rating"] == 2 for p in payloads)
    with pytest.raises(ValueError):
        Review("r2", "review text", "store", rating=6)


def test_store_documents_are_assigned_by_subfolder(tmp_path):
    (tmp_path / "store-a").mkdir()
    (tmp_path / "store-b" / "extra").mkdir(parents=True)
    (tmp_path / "common.md").write_text("공통 안내", encoding="utf-8")
    (tmp_path / "store-a" / "menu.md").write_text("떡볶이 메뉴", encoding="utf-8")
    (tmp_path / "store-b" / "extra" / "hours.txt").write_text("영업시간", encoding="utf-8")
    owners = {d.text: d.app_id for d in load_documents(tmp_path, "default")}
    assert owners == {"공통 안내": "default", "떡볶이 메뉴": "store-a", "영업시간": "store-b"}


class LargeModel:
    """Test-only generator standing in for a large API model."""

    identity = {"backend": "test-only-large"}

    def complete(self, messages, max_tokens):
        return "improved reply from large model"


def test_experiment_scores_comparison_models_with_base_prompt_file(tmp_path):
    base = tmp_path / "base.txt"
    base.write_text("매장 리뷰 답글 기본 프롬프트\n", encoding="utf-8")
    config = Config(
        train_size=4,
        validation_size=2,
        test_size=2,
        knowledge_size=8,
        max_iterations=1,
        threshold=1,
        base_prompt_file=str(base),
        compare_models=[{"backend": "anthropic", "model": "large"}],
        output_dir=str(tmp_path / "out"),
    )
    splits = split_reviews(reviews(), config)
    docs = knowledge_documents(splits["knowledge"])
    rag = HybridRetriever(docs, TestEmbeddings(), tmp_path / "rag")
    llm = ScriptedLLM()
    prepared = (splits, docs, {"source": "test fixture"})
    result = run_experiment(config, prepared, (llm, llm, llm), rag, compare_llms=(LargeModel(),))
    metrics = result["metrics"]
    assert result["optimization"].history[0]["prompt"] == "매장 리뷰 답글 기본 프롬프트"
    (comparison,) = metrics["comparisons"]
    assert comparison["model"] == "large" and comparison["count"] == 2
    assert comparison["overall"] > metrics["baseline"]["overall"]
    saved = json.loads((tmp_path / "out" / "test_predictions.json").read_text(encoding="utf-8"))
    assert saved["comparisons"][0]["predictions"][0]["prediction"] == "improved reply from large model"
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["base_prompt"] == "매장 리뷰 답글 기본 프롬프트"
    table = score_table(metrics, config)
    assert "anthropic:large" in table and "optimized" in table
    with pytest.raises(ValueError, match="compare_llms"):
        run_experiment(config, prepared, (llm, llm, llm), rag, compare_llms=())
