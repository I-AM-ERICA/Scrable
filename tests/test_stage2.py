"""Stage 2: synthetic training reviews, preference pairs, LoRA DPO and held-out evaluation."""

import json
import math
from types import SimpleNamespace

import pytest

import scrable.llm as llm_module
from scrable import Config
from scrable.data import split_reviews
from scrable.dpo import DPOConfig, build_pairs, dpo_loss, reselect_pairs, select_pair
from scrable.evaluate import evaluate, load_runs, paired_bootstrap
from scrable.rag import HybridRetriever
from scrable.synth_reviews import RATING_COUNTS, generate, plan
from scrable.types import CRITERIA, Document, Review
from tests.helpers import ScriptedLLM, TestEmbeddings

BASE = {
    "judge_backend": "anthropic",
    "judge_model": "judge",
    "custom_data": "unused.jsonl",
    "documents_dir": "docs",
    "knowledge_size": 0,
}


def candidate(source, text, score, accuracy=3):
    return {"source": source, "text": text, "score": score, "scores": {"accuracy": accuracy}}


def test_select_pair_prefers_best_answer_and_rejects_own_worst():
    cands = [
        candidate("policy", "a", 0.4),
        candidate("policy", "b", 0.7),
        candidate("teacher:anthropic:big", "c", 0.9),
    ]
    (chosen, rejected), reason = select_pair(cands, 0.15, 0.6)
    assert (chosen["text"], rejected["text"], reason) == ("c", "a", None)
    # Ties go to the small model's own answer (on-policy chosen).
    cands[1]["score"] = 0.9
    cands[1]["scores"]["accuracy"] = 3
    assert select_pair(cands, 0.15, 0.6)[0][0]["text"] == "b"


@pytest.mark.parametrize(
    "cands, reason",
    [
        ([candidate("teacher:x:y", "c", 0.9)], "no_policy_answer"),
        ([candidate("policy", "a", 0.9)], "single_answer"),
        ([candidate("policy", "a", 0.3), candidate("policy", "b", 0.5)], "chosen_too_low"),
        ([candidate("policy", "a", 0.7), candidate("policy", "b", 0.8)], "margin_too_small"),
    ],
)
def test_select_pair_skips_weak_signal(cands, reason):
    assert select_pair(cands, 0.15, 0.6) == (None, reason)


def scored(source, text, score, accuracy, specificity):
    return {
        "source": source,
        "text": text,
        "score": score,
        "scores": {"accuracy": accuracy, "specificity": specificity},
    }


def test_select_pair_generic_rejected_isolates_specificity():
    cands = [
        scored("policy", "wrong", 0.3, accuracy=2, specificity=2),
        scored("policy", "safe but vague", 0.6, accuracy=5, specificity=2),
        scored("teacher:x:y", "specific", 0.95, accuracy=5, specificity=4),
    ]
    assert select_pair(cands, 0.15, 0.6)[0][1]["text"] == "wrong"  # Default: the worst answer.
    (chosen, rejected), _ = select_pair(cands, 0.15, 0.6, rejected_strategy="generic")
    assert (chosen["text"], rejected["text"]) == ("specific", "safe but vague")
    no_generic = [cands[0], cands[2]]
    assert select_pair(no_generic, 0.15, 0.6, rejected_strategy="generic") == (None, "no_generic_answer")
    fallback = select_pair(no_generic, 0.15, 0.6, rejected_strategy="generic_or_lowest")[0]
    assert fallback[1]["text"] == "wrong"
    assert select_pair(cands, 0.15, 0.6, min_chosen_specificity=5) == (None, "chosen_not_specific")
    assert select_pair(cands, 0.15, 0.6, rejected_strategy="generic", min_specificity_gap=3) == (
        None,
        "specificity_gap_too_small",
    )


def test_dpo_config_validates_and_derives_paths(tmp_path):
    dcfg = DPOConfig(base={**BASE, "output_dir": str(tmp_path)})
    assert dcfg.pairs_file == tmp_path / "pairs.jsonl" and dcfg.adapter_dir == tmp_path / "adapter"
    for bad in (
        {"samples_per_review": 1},  # One answer and no teacher: nothing to compare.
        {"base": {**BASE, "backend": "anthropic", "model": "x"}},
        {"base": {**BASE, "custom_data": None}},
        {"teacher_models": [{"backend": "anthropic"}]},
        {"min_margin": 2},
        {"rejected_strategy": "random"},
        {"min_chosen_specificity": 6},
    ):
        with pytest.raises(ValueError):
            DPOConfig(**{"base": BASE, **bad})


def test_response_adapter_only_reaches_the_generator(monkeypatch, tmp_path):
    built = []

    def fake_backend(backend, model, config, adapter=None):
        built.append((backend, model, adapter))
        return SimpleNamespace(identity={"backend": backend, "model": model, "adapter": adapter})

    monkeypatch.setattr(llm_module, "make_backend", fake_backend)
    config = Config(response_adapter="adapter-dir", judge_model=None, output_dir=str(tmp_path))
    response, judge, _ = llm_module.build_llms(config)
    assert response is not judge  # Same base model, but the judge must not get the adapter.
    assert built == [("local", config.model, "adapter-dir"), ("local", config.model, None)]
    with pytest.raises(ValueError):
        Config(backend="anthropic", model="x", response_adapter="adapter-dir")


class SamplingLLM:
    """Test-only small model: answers carry a quality marker the fixture judge reads."""

    identity = {"backend": "test-only-sampler"}

    def __init__(self, texts):
        self.texts = texts
        self.calls = []

    def sample(self, messages, max_tokens, n, temperature, top_p, seed):
        self.calls.append((n, temperature, top_p, seed))
        return self.texts[:n]


class MarkerJudge:
    """Test-only judge: score 5 for 'good', 1 for 'bad', else 3."""

    identity = {"backend": "test-only-judge"}

    def complete(self, messages, max_tokens):
        prediction = json.loads(messages[-1]["content"])["prediction"]
        score = 5 if "good" in prediction else 1 if "bad" in prediction else 3
        return json.dumps({"score": score, "reason": "fixture", "suggestions": []})


class Teacher:
    identity = {"backend": "test-only-teacher"}

    def complete(self, messages, max_tokens):
        return "good teacher answer"


def test_build_pairs_writes_teacher_chosen_policy_rejected(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("학습용 프롬프트", encoding="utf-8")
    dcfg = DPOConfig(
        base={
            **BASE,
            "base_prompt_file": str(prompt),
            "output_dir": str(tmp_path / "out"),
            "judge_workers": 4,
        },
        samples_per_review=3,
        teacher_models=[{"backend": "anthropic", "model": "big"}],
    )
    rows = [Review(f"r{i}", "배달이 늦었어요", "store", rating=2) for i in range(3)]
    rag = HybridRetriever([Document("d", "배달 안내", "store", "doc")], TestEmbeddings(), tmp_path / "rag")
    policy = SamplingLLM(["bad answer", "okay answer", "bad answer"])
    summary = build_pairs(dcfg, rows, rag, policy, MarkerJudge(), (Teacher(),))
    pairs = [json.loads(x) for x in dcfg.pairs_file.read_text(encoding="utf-8").splitlines()]
    assert summary["pairs"] == len(pairs) == 3 and summary["candidates"] == 9  # Duplicate sample dropped.
    pair = pairs[0]
    assert pair["chosen"] == "good teacher answer" and pair["rejected"] == "bad answer"
    assert pair["chosen_source"] == "teacher:anthropic:big" and pair["chosen_score"] == 1.0
    assert pair["messages"][0]["content"].startswith("학습용 프롬프트")
    assert json.loads(pair["messages"][1]["content"])["rating"] == 2
    assert policy.calls[0] == (3, 0.8, 0.95, 42)
    lines = (tmp_path / "out" / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    assert set(json.loads(lines[0])["candidates"][0]["scores"]) == set(CRITERIA)
    assert json.loads(lines[0])["messages"] == pair["messages"]  # Enough to re-select pairs later.

    # Re-selecting with the same options reproduces the pairs without any model call.
    same = DPOConfig(**{**dcfg.to_dict(), "base": {**dcfg.base, "output_dir": str(tmp_path / "again")}})
    summary = reselect_pairs(same, tmp_path / "out")
    again = [json.loads(x) for x in same.pairs_file.read_text(encoding="utf-8").splitlines()]
    assert again == pairs and summary["pairs"] == 3
    strict = DPOConfig(
        **{
            **dcfg.to_dict(),
            "base": {**dcfg.base, "output_dir": str(tmp_path / "strict")},
            "min_chosen_specificity": 5,
            "rejected_strategy": "generic",
        }
    )
    summary = reselect_pairs(strict, tmp_path / "out")
    assert summary["pairs"] == 0 and summary["skipped"] == {"no_generic_answer": 3}


def test_dpo_loss_matches_formula():
    torch = pytest.importorskip("torch")
    zero = torch.zeros(1)
    loss, stats = dpo_loss(zero, zero, zero, zero, beta=0.1)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6) and stats["reward_margin"] == 0
    loss, stats = dpo_loss(torch.tensor([1.0]), torch.tensor([-1.0]), zero, zero, beta=0.5)
    assert math.isclose(loss.item(), math.log1p(math.exp(-1.0)), rel_tol=1e-6)
    assert stats["reward_accuracy"] == 1.0
    with_sft, _ = dpo_loss(torch.tensor([-4.0]), zero, zero, zero, 0.1, 0.5, torch.tensor([2]))
    plain, _ = dpo_loss(torch.tensor([-4.0]), zero, zero, zero, 0.1)
    assert math.isclose(with_sft.item() - plain.item(), 0.5 * 2.0, rel_tol=1e-6)


class CharTokenizer:
    """Test-only tokenizer: one token per character (mod a tiny vocabulary)."""

    pad_token_id = 0
    eos_token_id = 1
    eos_token = "|"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + ord(ch) % 60 for ch in text]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


def tiny_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    return LlamaForCausalLM(config)


def test_completion_logps_ignores_prompt_and_padding():
    torch = pytest.importorskip("torch")
    from scrable.dpo import completion_logps

    model = tiny_model().eval()
    prompt, short, long = [5, 6, 7], [8], [9, 10, 11]
    with torch.no_grad():
        values, counts = completion_logps(model, prompt, [short, long], 0, "cpu")
        logits = model(input_ids=torch.tensor([prompt + long])).logits[0].log_softmax(-1)
    expected = sum(logits[len(prompt) - 1 + k, tok].item() for k, tok in enumerate(long))
    assert counts.tolist() == [1, 3]
    assert math.isclose(values[1].item(), expected, rel_tol=1e-4)


def test_train_dpo_learns_preference_and_saves_adapter(tmp_path):
    pytest.importorskip("peft")
    from scrable.dpo import train_dpo

    dcfg = DPOConfig(
        base={**BASE, "model": "tiny-test-model", "output_dir": str(tmp_path)},
        learning_rate=5e-3,
        epochs=3,
        grad_accum=2,
        sft_weight=0.0,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        eval_fraction=0.0,
    )
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "review"}]
    pairs = [{"messages": messages, "chosen": "thanks", "rejected": "you??"} for _ in range(8)]
    log = train_dpo(dcfg, pairs, tiny_model(), CharTokenizer(), "cpu")
    assert log["trained_pairs"] == 8 and log["skipped_too_long"] == 0
    assert log["steps"][-1]["reward_margin"] > log["steps"][0]["reward_margin"]
    assert log["steps"][-1]["reward_accuracy"] == 1.0
    assert (dcfg.adapter_dir / "adapter_config.json").exists()
    assert json.loads((tmp_path / "train_log.json").read_text(encoding="utf-8"))["pairs"] == 8


def test_paired_bootstrap_and_reference_runs(tmp_path):
    result = paired_bootstrap([1.0, 1.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5])
    assert result["mean"] == 0.5 and (result["wins"], result["losses"], result["ties"]) == (2, 0, 2)
    assert 0 <= result["low"] <= 0.5 <= result["high"] <= 1
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [])


def test_evaluate_scores_test_split_and_compares_with_stage1(tmp_path):
    rows = [Review(str(i), f"Food delivery review number {i}", "app", rating=3) for i in range(12)]
    config = Config(
        train_size=4,
        validation_size=2,
        test_size=4,
        knowledge_size=0,
        documents_dir="docs",
        output_dir=str(tmp_path / "eval"),
    )
    splits = split_reviews(rows, config)
    rag = HybridRetriever([Document("d", "delivery facts", "app", "doc")], TestEmbeddings(), tmp_path / "rag")
    test_ids = [r.id for r in splits["test"]]

    def feedback(i, score):
        scores = dict.fromkeys(CRITERIA, score)
        return {"review_id": i, "scores": scores, "score": (score - 1) / 4}

    reference = tmp_path / "stage1.json"
    reference.write_text(
        json.dumps(
            {
                "baseline": [feedback(i, 2) for i in test_ids],
                "optimized": [feedback(i, 2) for i in test_ids],
                "comparisons": [{"model": "big", "predictions": [feedback(i, 4) for i in test_ids]}],
            }
        ),
        encoding="utf-8",
    )
    assert set(load_runs(reference)) == {"stage1:baseline", "stage1:optimized", "big:base"}
    llm = ScriptedLLM()  # Base prompt replies 'baseline reply', which the fixture judge scores 2.
    metrics = evaluate(config, "dpo", str(reference), (splits, [], {}), (llm, llm), rag)
    assert metrics["summary"]["count"] == 4
    assert metrics["comparisons"]["stage1:optimized"]["overall"]["mean"] == 0
    assert metrics["comparisons"]["big:base"]["overall"]["mean"] == -0.5
    saved = json.loads((tmp_path / "eval" / "test_predictions.json").read_text(encoding="utf-8"))
    assert saved["label"] == "dpo" and len(saved["predictions"]) == 4
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"label": "x", "predictions": [feedback("zzz", 3)]}), encoding="utf-8")
    with pytest.raises(ValueError, match="different test reviews"):
        evaluate(config, "dpo", str(other), (splits, [], {}), (llm, llm), rag)
    # The dev split (train + validation) is for choosing variants without touching the test set.
    metrics = evaluate(config, "dev-run", None, (splits, [], {}), (llm, llm), rag, split="dev")
    saved = json.loads((tmp_path / "eval" / "dev_predictions.json").read_text(encoding="utf-8"))
    dev_ids = {r.id for r in splits["train"] + splits["validation"]}
    assert metrics["split"] == "dev" and {p["review_id"] for p in saved["predictions"]} == dev_ids
    assert not dev_ids & set(test_ids)


class ReviewWriter:
    """Test-only API model writing the requested number of distinct reviews."""

    identity = {"backend": "test-only-writer"}

    def __init__(self):
        self.calls = 0

    def complete(self, messages, max_tokens):
        import hashlib

        self.calls += 1
        payload = json.loads(messages[-1]["content"])
        count = len(payload["reviews_to_write"])
        reviews = [hashlib.sha256(f"{self.calls}-{i}".encode()).hexdigest()[:20] for i in range(count)]
        reviews[0] = "떡볶이 국물이 진짜 맛있어요"  # Copies a review that must be avoided.
        return json.dumps({"reviews": reviews}, ensure_ascii=False)


def test_synthetic_reviews_follow_plan_and_skip_copies():
    batches = plan(["store-a", "store-b"])
    per_store = {}
    for store, rating, specs in batches:
        assert len(specs) <= 8
        per_store.setdefault(store, {}).setdefault(rating, 0)
        per_store[store][rating] += len(specs)
    assert per_store["store-a"] == RATING_COUNTS == per_store["store-b"]
    assert plan(["store-a"]) == plan(["store-a"])  # Deterministic for the same seed.

    docs = [Document("d", "가게 정보", "store-a", "info.md")]
    rows, dropped = generate(
        ReviewWriter(), docs, plan(["store-a"]), avoid=["떡볶이 국물이 진짜 맛있어요 ㅠㅠ"]
    )
    assert dropped >= 1 and all("떡볶이" not in r["review"] for r in rows)
    assert len({r["id"] for r in rows}) == len(rows) and all(r["app_id"] == "store-a" for r in rows)
    assert {r["rating"] for r in rows} == set(RATING_COUNTS)


def test_train_dpo_with_dora_saves_a_dora_adapter(tmp_path):
    pytest.importorskip("peft")
    from scrable.dpo import train_dpo

    dcfg = DPOConfig(
        base={**BASE, "model": "tiny-test-model", "output_dir": str(tmp_path)},
        epochs=1,
        grad_accum=2,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        eval_fraction=0.0,
        use_dora=True,
        gradient_checkpointing=True,  # as in configs/stage4_photo_dora.json
    )
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "review"}]
    pairs = [{"messages": messages, "chosen": "thanks", "rejected": "you??"} for _ in range(4)]
    train_dpo(dcfg, pairs, tiny_model(), CharTokenizer(), "cpu")
    saved = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["use_dora"] is True and saved["r"] == 4
