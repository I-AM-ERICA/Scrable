"""Email auto-reply test: synthetic emails, email channel wording and the all split."""

import json

import pytest

from scrable import Config
from scrable.evaluate import evaluate, load_references
from scrable.pipeline import generation_messages
from scrable.prompts import JUDGE_SYSTEM, judge_system, rubric
from scrable.rag import HybridRetriever
from scrable.synth_emails import KINDS, generate, kind_of, plan
from scrable.types import Document, Review
from tests.helpers import ScriptedLLM, TestEmbeddings


class EmailWriter:
    """Test-only API model writing the requested number of emails."""

    identity = {"backend": "test-only-email-writer"}

    def __init__(self, repeat=None):
        self.calls, self.payloads, self.repeat = 0, [], repeat

    def complete(self, messages, max_tokens):
        import hashlib

        self.calls += 1
        payload = json.loads(messages[-1]["content"])
        self.payloads.append(payload)
        emails = [
            {"subject": "문의", "body": hashlib.sha256(f"{self.calls}-{i}".encode()).hexdigest()[:30]}
            for i, _ in enumerate(payload["emails_to_write"])
        ]
        if self.repeat:
            emails[0]["body"] = self.repeat  # A copy of an email that must be avoided.
        return json.dumps({"emails": emails}, ensure_ascii=False)


def test_synthetic_emails_cover_every_kind_per_store():
    batches = plan(["store-a", "store-b"])
    assert plan(["store-a", "store-b"]) == batches
    docs = [Document("d", "가게 정보", s, "info.md") for s in ("store-a", "store-b")]
    writer = EmailWriter()
    rows, dropped = generate(writer, docs, batches)
    assert dropped == 0 and len(rows) == 2 * len(KINDS) and len({r["id"] for r in rows}) == len(rows)
    assert {kind_of(r["id"]) for r in rows if r["app_id"] == "store-a"} == set(KINDS)
    assert all(r["review"].startswith("제목: ") and Review(**r) for r in rows)
    # The test set's requests (and so its ids and cache) carry no style.
    assert rows[0]["id"] == "email-store-a-hours" and "style" not in writer.payloads[0]["emails_to_write"][0]
    with pytest.raises(ValueError, match="No store documents"):
        generate(EmailWriter(), docs[:1], plan(["store-b"]))


def test_training_emails_repeat_kinds_in_varied_tones_and_skip_copies():
    docs = [Document("d", "가게 정보", "store-a", "info.md")]
    writer = EmailWriter(repeat="떡볶이 순한맛 되나요? 알려 주세요")
    rows, dropped = generate(
        writer, docs, plan(["store-a"], per_kind=3), avoid=["제목: 문의\n\n떡볶이 순한맛 되나요? 알려 주세요"]
    )
    assert dropped == 3 * 2 and len(rows) == 3 * len(KINDS) - dropped  # One copy per request.
    assert len({r["id"] for r in rows}) == len(rows)
    first = {"hours", "allergy"}  # Each request's first email is the copy, so these are dropped.
    assert {kind_of(r["id"]) for r in rows} == set(KINDS) - first and any(
        r["id"].endswith("-03") for r in rows
    )
    assert len({p["emails_to_write"][0]["style"] for p in writer.payloads}) == 3


def test_email_channel_changes_inputs_and_judge_wording_only():
    review = Review("e", "제목: 문의\n\n영업시간이 궁금해요", "store")
    review_input = json.loads(generation_messages("p", review, [])[1]["content"])
    email_input = json.loads(generation_messages("p", review, [], "email")[1]["content"])
    assert "review" in review_input and "email" in email_input and "review" not in email_input
    assert judge_system() == JUDGE_SYSTEM and "메일" in judge_system("email")
    assert rubric("accuracy") != rubric("accuracy", "email") and rubric("specificity") == rubric(
        "specificity", "email"
    )
    with pytest.raises(ValueError, match="channel"):
        Config(knowledge_size=0, documents_dir="d", channel="chat")


def test_all_split_scores_every_email_with_several_references(tmp_path):
    data = tmp_path / "emails.jsonl"
    rows = [
        {"id": f"email-app-{k}", "app_id": "app", "review": f"제목: {k}\n\n문의"} for k in ("hours", "menu")
    ]
    data.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    docs = tmp_path / "docs" / "app"
    docs.mkdir(parents=True)
    (docs / "info.md").write_text("영업시간 11시", encoding="utf-8")
    config = Config(
        knowledge_size=0,
        documents_dir=str(tmp_path / "docs"),
        custom_data=str(data),
        channel="email",
        output_dir=str(tmp_path / "out"),
    )
    rag = HybridRetriever([Document("d", "영업시간 11시", "app", "doc")], TestEmbeddings(), tmp_path / "rag")
    llm = ScriptedLLM()
    metrics = evaluate(config, "lora", None, None, (llm, llm), rag, split="all")
    assert metrics["summary"]["count"] == 2 and metrics["channel"] == "email"
    judged = [json.loads(m[-1]["content"]) for m in llm.messages if "criterion" in m[-1]["content"]]
    assert judged and all("email" in p and "review" not in p for p in judged)
    first = tmp_path / "out" / "all_predictions.json"
    second = tmp_path / "second.json"
    second.write_text(
        json.dumps({"label": "haiku", **json.loads(first.read_text(encoding="utf-8")) | {"label": "haiku"}}),
        encoding="utf-8",
    )
    assert set(load_references([str(first), str(second)])) == {"lora", "haiku"}
    metrics = evaluate(config, "base", [str(first), str(second)], None, (llm, llm), rag, split="all")
    assert set(metrics["comparisons"]) == {"lora", "haiku"}


def test_email_pairs_use_email_input_and_train_with_review_pairs(tmp_path):
    from scrable.dpo import DPOConfig, build_pairs, load_training_pairs
    from tests.test_stage2 import BASE, MarkerJudge, SamplingLLM, Teacher

    review_pairs = tmp_path / "review_pairs.jsonl"
    review_pairs.write_text(
        json.dumps({"review_id": "r1", "messages": [], "chosen": "a", "rejected": "b"}) + "\n",
        encoding="utf-8",
    )
    dcfg = DPOConfig(
        base={**BASE, "channel": "email", "output_dir": str(tmp_path / "out")},
        samples_per_review=2,
        teacher_models=[{"backend": "anthropic", "model": "big"}],
        extra_pairs=[str(review_pairs)],
    )
    rows = [Review("email-store-hours", "제목: 문의\n\n몇 시까지 하나요?", "store")]
    rag = HybridRetriever(
        [Document("d", "영업시간 안내", "store", "doc")], TestEmbeddings(), tmp_path / "rag"
    )
    judge = MarkerJudge()
    build_pairs(dcfg, rows, rag, SamplingLLM(["bad answer", "okay answer"]), judge, (Teacher(),))
    pairs = load_training_pairs(dcfg)
    assert [p["review_id"] for p in pairs] == ["email-store-hours", "r1"]
    assert "email" in json.loads(pairs[0]["messages"][1]["content"])
    twice = DPOConfig(**{**dcfg.to_dict(), "extra_pairs": [str(review_pairs), str(review_pairs)]})
    with pytest.raises(ValueError, match="Duplicate review_id"):
        load_training_pairs(twice)
    with pytest.raises(ValueError, match="extra_pairs"):
        DPOConfig(**{**dcfg.to_dict(), "extra_pairs": "outputs/stage2/pairs.jsonl"})
