import json

import pytest

from scrable import Config
from scrable.data import knowledge_documents, split_reviews
from scrable.experiment import run_experiment
from scrable.optimizer import PromptOptimizer
from scrable.pipeline import ScoredResponseGenerator
from scrable.rag import HybridRetriever
from scrable.types import CRITERIA, Document, Feedback
from tests.helpers import ScriptedLLM, TestEmbeddings, reviews


def test_retrieval_filters_apps_and_reloads_cache(tmp_path):
    docs = [Document("a", "food delivery", "app-a", "one"), Document("b", "food refund", "app-b", "two")]
    first = HybridRetriever(docs, TestEmbeddings(), tmp_path)
    assert [d.id for d in first.retrieve("food refund", "app-a")] == ["a"]
    assert first.retrieve("food", "missing") == []
    second = HybridRetriever(docs, TestEmbeddings(), tmp_path)
    assert first.fingerprint == second.fingerprint
    assert first.retrieve("food", "app-a") == second.retrieve("food", "app-a")


def test_reference_only_reaches_accuracy_judge(tmp_path):
    llm = ScriptedLLM()
    config = Config()
    rag = HybridRetriever(
        [Document("d", "Relevant documented facts", "app", "manual")], TestEmbeddings(), tmp_path
    )
    scorer = ScoredResponseGenerator(llm, llm, rag, config)
    row = reviews(1)[0]
    prediction, chunks = scorer.generate("base prompt", row)
    scorer.judge(row, prediction, chunks)
    assert "REFERENCE-ONLY" not in json.dumps(llm.messages[0])
    for messages in llm.messages[1:]:
        payload = json.loads(messages[-1]["content"])
        assert ("expert_response" in payload) == (payload["criterion"] == "accuracy")
        # Store documents reach specificity and accuracy, so documented facts are not "invented".
        has_documents = "retrieved_documents" in payload
        assert has_documents == (payload["criterion"] in {"specificity", "accuracy"})
        if has_documents:
            assert payload["retrieved_documents"][0]["text"] == "Relevant documented facts"


def test_full_experiment_artifacts_and_test_isolation(tmp_path):
    config = Config(
        train_size=4,
        validation_size=2,
        test_size=2,
        knowledge_size=8,
        max_iterations=1,
        threshold=1,
        output_dir=str(tmp_path),
    )
    splits = split_reviews(reviews(), config)
    docs = knowledge_documents(splits["knowledge"])
    rag = HybridRetriever(docs, TestEmbeddings(), tmp_path / "rag")
    llm = ScriptedLLM()
    result = run_experiment(config, (splits, docs, {"source": "test fixture"}), (llm, llm, llm), rag)
    assert result["metrics"]["best_iteration"] == 1
    assert result["metrics"]["delta"]["overall"] > 0
    assert len(result["optimization"].history) == 2
    for name in ("manifest.json", "metrics.json", "history.json", "test_predictions.json", "best_prompt.txt"):
        assert (tmp_path / name).is_file()
    generation_inputs = [json.loads(m[-1]["content"]) for m in llm.messages]
    prompt_inputs = [p for p in generation_inputs if "current_prompt" in p]
    for payload in prompt_inputs:
        for test_review in splits["test"]:
            assert test_review.review not in json.dumps(payload)
    history = result["optimization"].history
    assert all(len(item["train_feedback"]) == 4 for item in history)
    knowledge_ids = {f"history-{r.id}" for r in splits["knowledge"]}
    assert all(set(item["context_ids"]) <= knowledge_ids for item in history[0]["train_feedback"])
    # Same output directory must not silently combine different datasets/configs.
    config.seed = 99
    with pytest.raises(ValueError, match="different experiment"):
        run_experiment(config, (splits, docs, {"source": "test fixture"}), (llm, llm, llm), rag)


def test_threshold_can_stop_at_baseline_and_zero_iterations_is_valid(tmp_path):
    class Scorer:
        def __call__(self, prompt, rows, description):
            return [
                Feedback(r.id, r.review, r.app_id, "reply", {k: 5 for k in CRITERIA}, [], {}) for r in rows
            ]

    def should_not_revise(*args):
        raise AssertionError("Already at threshold")

    config = Config(output_dir=str(tmp_path))
    result = PromptOptimizer(Scorer(), should_not_revise, config).run(reviews(1), reviews(3)[1:])
    assert result.stop_reason == "threshold" and len(result.history) == 1
    config.max_iterations = 0
    assert (
        len(PromptOptimizer(Scorer(), should_not_revise, config).run(reviews(1), reviews(3)[1:]).history) == 1
    )


def test_best_prompt_uses_validation_and_preserves_last_prompt(tmp_path):
    class Scorer:
        def __call__(self, prompt, rows, description):
            score = 5 if "train" in description else 4 if prompt == "base" else 2
            # Keep training below threshold until last iteration.
            if "train" in description:
                score = 3
            return [
                Feedback(r.id, r.review, r.app_id, "reply", {k: score for k in CRITERIA}, [], {})
                for r in rows
            ]

    config = Config(output_dir=str(tmp_path), max_iterations=1, threshold=1)
    result = PromptOptimizer(Scorer(), lambda *_: "worse revision", config).run(
        reviews(1), reviews(3)[1:], base_prompt="base"
    )
    assert result.best_prompt == "base"
    assert result.final_prompt == "worse revision"
    assert result.best_iteration == 0
