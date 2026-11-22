"""SCRABLE gRPC server: reply engine, internal key, health check and a real call on localhost."""

import json
import threading

import pytest

from scrable.serve import (
    Busy,
    ReplyEngine,
    ServeConfig,
    estimated_rating,
    keywords_of,
    sentiment_of,
    store_document,
)
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
        {"store_id": "99", "name": "새가게", "category": "카페"}, "그냥 그래요", 0, tone="친근하게"
    )
    payload = json.loads(llm.messages[-1][1]["content"])
    assert fallback["reply"].endswith("(docs=1)") and "rating" not in payload  # 0 and no cue: unknown
    assert fallback["sentiment"] == "neutral"
    assert "가게 이름: 새가게" in payload["retrieved_documents"][0]["text"]
    assert "답글 말투" not in llm.messages[-1][0]["content"]  # tone is off by default (untrained)
    toned = ReplyEngine(ServeConfig(base=BASE, use_tone=True), llm, FakeRetriever([]), "기본 프롬프트")
    toned.draft({"store_id": "99", "name": "새가게"}, "그냥 그래요", 0, tone="친근하게")
    assert llm.messages[-1][0]["content"].startswith("기본 프롬프트\n- 답글 말투: 친근하게")


def test_store_info_from_the_owner_becomes_the_store_document():
    llm = FakeLLM()
    eng = engine(llm, aliases={"7": "bunsik-01"})
    info = "메뉴: 떡볶이, 김밥, 순대\n영업시간 11:00~21:00 / 월요일 휴무\n누락은 매장 전화로 알려 주시면 확인 후 다시 보내 드립니다."
    eng.draft({"store_id": "7", "name": "우리분식", "category": "분식"}, "배달이 누락됐어요 전화 드렸어요", 2,
              store_info=info)
    docs = json.loads(llm.messages[-1][1]["content"])["retrieved_documents"]
    assert docs[0]["source"] == "사장님이 입력한 가게 정보"
    lines = docs[0]["text"].splitlines()
    assert lines[0].startswith("가게 정보") and "주문한 메뉴가 아닙니다" in lines[0]
    assert len(lines) == 3 and any("누락은 매장 전화" in line for line in lines)  # header + 2 chosen lines
    assert "가게 이름: 우리분식" in docs[-1]["text"]  # basic store fields still come along
    assert all("단무지" not in d["text"] for d in docs)  # the folder document is not used
    eng.draft({"store_id": "7"}, "김밥이 빠졌어요", 2, store_info="   ")  # blank: folder documents as before
    assert json.loads(llm.messages[-1][1]["content"])["retrieved_documents"][0]["id"] == "d1"


def test_menu_names_only_from_the_review_and_required_lines_are_appended():
    llm = FakeLLM()
    eng = engine(llm)
    info = ("메뉴: 떡볶이, 김밥, 순대\n누락은 매장 전화로 알려 주세요\n"
            "필수: 리뷰 이벤트 참여 감사드려요! 다음 주문 때 음료를 서비스로 드려요.")
    result = eng.draft({"store_id": "x"}, "떡볶이가 너무 매웠어요", 2, store_info=info)
    text = json.loads(llm.messages[-1][1]["content"])["retrieved_documents"][0]["text"]
    assert "떡볶이" in text and "김밥" not in text and "순대" not in text
    assert "리뷰 이벤트" not in text  # required sentences are not sent to the model
    assert result["reply"].endswith("\n\n리뷰 이벤트 참여 감사드려요! 다음 주문 때 음료를 서비스로 드려요.")
    eng.draft({"store_id": "x"}, "직원분이 친절해요", 5, store_info="메뉴: 떡볶이, 김밥")
    text = json.loads(llm.messages[-1][1]["content"])["retrieved_documents"][0]["text"]
    assert "떡볶이" not in text and "관련된 가게 정보 없음" in text


def test_long_store_info_is_ranked_for_the_review():
    from scrable.rag import rank_chunks

    chunks = [Document(f"c{i}", text, "s", "x") for i, text in enumerate(["영업시간 11시부터", "배달 누락 안내 전화", "주차 가능"])]
    assert [c.id for c in rank_chunks("배달이 누락됐어요 전화 주세요", chunks, None, 1)] == ["c1"]
    assert rank_chunks("아무거나", chunks[:2], None, 2) == chunks[:2]


def test_missing_rating_is_guessed_from_the_review_text():
    assert estimated_rating("원장님이 꼼꼼하게 잘 잘라 주셨어요") == 5
    assert estimated_rating("젤네일이 사흘 만에 들떴어요. 아쉽네요") == 2
    assert estimated_rating("맛있는데 배달이 늦었어요") == 3
    assert estimated_rating("포장해 갑니다") is None
    llm = FakeLLM()
    eng = engine(llm)
    result = eng.draft({"store_id": "bunsik-01"}, "단무지가 빠져서 왔어요", 0)
    assert result["sentiment"] == "negative"
    assert json.loads(llm.messages[-1][1]["content"])["rating"] == 2
    sent = eng.draft({"store_id": "bunsik-01"}, "단무지가 빠져서 왔어요", 4)  # a sent rating wins
    assert sent["sentiment"] == "positive" and json.loads(llm.messages[-1][1]["content"])["rating"] == 4


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
    with pytest.raises(ValueError, match="use_tone"):
        ServeConfig(base=BASE, use_tone="yes")


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
    with_info = call(server, KEY, review_text="배달이 늦었어요", rating=2, store_info="메뉴: 떡볶이\n배달 문의는 매장 전화")
    assert with_info.reply.endswith("(docs=1)")  # the owner's store info replaces the folder document
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
