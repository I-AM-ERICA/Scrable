"""SCRABLE gRPC server: reply engine, internal key, health check and a real call on localhost."""

import json
import threading

import pytest

from scrable.serve import Busy, ReplyEngine, ServeConfig, keywords_of, sentiment_of, store_document
from scrable.types import Document

grpc = pytest.importorskip("grpc")
pytest.importorskip("grpc_health")
pytest.importorskip("grpc_tools")

BASE = {"documents_dir": "docs", "knowledge_size": 0, "base_prompt_file": None}
KEY = "test-internal-key"


class FakeLLM:
    """Test-only model: echoes how many store documents it was given."""

    def __init__(self, gate=None):
        self.messages, self.gate = [], gate

    def complete(self, messages, max_tokens):
        if self.gate:
            self.gate.wait(5)
        self.messages.append(messages)
        docs = json.loads(messages[1]["content"])["retrieved_documents"]
        return f"고객님, 감사합니다. (docs={len(docs)})"


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query, app_id, top_k):
        return [d for d in self.docs if d.app_id == app_id][:top_k]


def engine(llm=None, aliases=None, max_waiting=4):
    scfg = ServeConfig(base=BASE, store_aliases=aliases or {}, max_waiting=max_waiting)
    docs = [Document("d1", "단무지는 1개 기본 제공", "bunsik-01", "info.md")]
    return ReplyEngine(scfg, llm or FakeLLM(), FakeRetriever(docs), "기본 프롬프트")


def test_sentiment_keywords_and_store_fallback_document():
    assert [sentiment_of(r) for r in (5, 4, 3, 0, 2, 1)] == ["positive"] * 2 + ["neutral"] * 2 + [
        "negative"
    ] * 2
    assert keywords_of("배달이 한참 늦어서 다 식어서 왔어요") == ["배달 지연", "음식 온도"]
    assert keywords_of("그냥 그래요") == []
    assert store_document({"name": "예시분식", "category": "분식", "region": "", "address": ""}).endswith(
        "- 가게 이름: 예시분식\n- 업종: 분식"
    )
    assert store_document({"name": " "}) is None


def test_engine_uses_store_documents_alias_or_store_context():
    llm = FakeLLM()
    eng = engine(llm, aliases={"7": "bunsik-01"})
    by_alias = eng.draft({"store_id": "7"}, "단무지가 없어요", 2)
    assert by_alias["reply"].endswith("(docs=1)") and by_alias["sentiment"] == "negative"
    assert json.loads(llm.messages[-1][1]["content"])["retrieved_documents"][0]["id"] == "d1"
    fallback = eng.draft(
        {"store_id": "99", "name": "새가게", "category": "카페"}, "맛있어요", 0, tone="친근하게"
    )
    payload = json.loads(llm.messages[-1][1]["content"])
    assert fallback["reply"].endswith("(docs=1)") and "rating" not in payload  # 0 = unknown rating
    assert "가게 이름: 새가게" in payload["retrieved_documents"][0]["text"]
    assert llm.messages[-1][0]["content"].startswith("기본 프롬프트\n- 답글 말투: 친근하게")


class FakeDescriber:
    """Test-only vision model: records (photo, menu) for each description."""

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def describe(self, path, menu=None):
        self.calls.append((path, menu))
        if self.fail:
            raise ValueError("unreadable photo")
        return "떡볶이 한 접시가 보입니다"


def test_engine_describes_photo_with_store_menu_when_photo_context_is_on():
    llm, describer = FakeLLM(), FakeDescriber()
    docs = [Document("d1", "단무지는 1개 기본 제공", "bunsik-01", "info.md")]
    menus = {"bunsik-01": "떡볶이, 김밥"}
    on = ReplyEngine(
        ServeConfig(base={**BASE, "photo_context": True}, store_aliases={"7": "bunsik-01"}),
        llm, FakeRetriever(docs), "p", describer, menus,
    )
    on.draft({"store_id": "7"}, "이거 보세요", 2, image="photo.jpg")
    assert describer.calls == [("photo.jpg", "떡볶이, 김밥")]  # menu of the aliased documents folder
    payload = json.loads(llm.messages[-1][1]["content"])
    assert payload["photo_description"] == "떡볶이 한 접시가 보입니다"
    on.draft({"store_id": "99", "name": "새가게"}, "이거 보세요", 2, image="photo.jpg")
    assert describer.calls[-1] == ("photo.jpg", None)  # store without documents: no menu
    on.draft({"store_id": "7"}, "맛있어요", 5)  # no photo: no description, training-time input
    assert len(describer.calls) == 2 and "photo_description" not in json.loads(llm.messages[-1][1]["content"])
    off = ReplyEngine(ServeConfig(base=BASE), llm, FakeRetriever(docs), "p", describer, menus)
    off.draft({"store_id": "bunsik-01"}, "이거 보세요", 2, image="photo.jpg")
    assert describer.calls[-1] == ("photo.jpg", None)  # photo_context off: photo only


def test_engine_replies_without_photo_when_description_fails():
    llm = FakeLLM()
    eng = ReplyEngine(ServeConfig(base=BASE), llm, FakeRetriever([]), "p", FakeDescriber(fail=True))
    assert eng.draft({"store_id": "x", "name": "새가게"}, "이거 보세요", 1, image="photo.jpg")["reply"]
    assert "photo_description" not in json.loads(llm.messages[-1][1]["content"])


def test_engine_refuses_when_too_many_calls_wait():
    gate = threading.Event()
    eng = engine(FakeLLM(gate), max_waiting=1)
    worker = threading.Thread(target=eng.draft, args=({"store_id": "bunsik-01"}, "늦었어요", 2))
    worker.start()
    try:
        with pytest.raises(Busy):
            for _ in range(50):  # Until the first call holds the only slot.
                eng.draft({"store_id": "bunsik-01"}, "또 늦었어요", 2)
    finally:
        gate.set()
        worker.join()


def test_serve_config_validation():
    with pytest.raises(ValueError, match="backend must be local"):
        ServeConfig(base={**BASE, "backend": "anthropic", "model": "m"})
    with pytest.raises(ValueError, match="store_aliases"):
        ServeConfig(base=BASE, store_aliases={"1": 2})


@pytest.fixture
def server():
    from scrable.serve import build_server, make_servicer

    srv, port = build_server(make_servicer(engine()), KEY, "127.0.0.1", 0, 2)
    srv.start()
    yield port
    srv.stop(None)


def call(port, key, **fields):
    from america.ai.v1 import common_pb2

    from scrable.grpc_gen import customer_modules

    pb, pb_grpc = customer_modules()
    request = pb.ReviewReplyRequest(store=common_pb2.StoreContext(store_id="bunsik-01"), **fields)
    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
        metadata = (("x-internal-key", key),) if key else ()
        return pb_grpc.CustomerAssistantStub(channel).DraftReply(request, metadata=metadata, timeout=10)


def test_grpc_round_trip_key_health_and_validation(server):
    from grpc_health.v1 import health_pb2, health_pb2_grpc

    result = call(server, KEY, review_text="배달이 늦어서 식었어요", rating=2)
    assert result.reply.endswith("(docs=1)") and result.sentiment == "negative"
    assert list(result.keywords) == ["배달 지연", "음식 온도"]
    for key in ("wrong-key", None):
        with pytest.raises(grpc.RpcError) as denied:
            call(server, key, review_text="맛있어요", rating=5)
        assert denied.value.code() == grpc.StatusCode.UNAUTHENTICATED
    with pytest.raises(grpc.RpcError) as empty:
        call(server, KEY, review_text="   ", rating=5)
    assert empty.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    with pytest.raises(grpc.RpcError) as bad_rating:
        call(server, KEY, review_text="맛있어요", rating=9)
    assert bad_rating.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    with grpc.insecure_channel(
        f"127.0.0.1:{server}"
    ) as channel:  # Health needs no key, as the backend polls it.
        status = health_pb2_grpc.HealthStub(channel).Check(health_pb2.HealthCheckRequest(), timeout=5).status
    assert status == health_pb2.HealthCheckResponse.SERVING
