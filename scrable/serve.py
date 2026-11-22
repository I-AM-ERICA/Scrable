"""gRPC server for the backend's review-reply call (CustomerAssistant.DraftReply, port 50053).

    read -rs AI_INTERNAL_KEY && export AI_INTERNAL_KEY
    python -m scrable.serve --config configs/serve.json

Implements america.ai.v1.CustomerAssistant from am-erica-backend's customer.proto (copied to
scrable/proto) with the stage-2 model (1.5B + LoRA, the same prompt and store-document retrieval
as the experiments), plus the standard gRPC health service the backend polls. DraftReply calls
must carry metadata x-internal-key equal to AI_INTERNAL_KEY (the backend's ai.grpc.internal-key).

Store documents: the backend sends StoreContext (id, name, category, region, address). Documents
come from documents_dir/<store_id>/ (or the folder store_aliases maps the id to); a store without
documents gets one short document built from StoreContext, so replies stay generic but grounded.
sentiment comes from the star rating (same rule as the backend's mock), keywords from a fixed list
of review topics. Replies are generated one at a time on the GPU; past max_waiting queued calls the
server answers RESOURCE_EXHAUSTED (the backend shows "AI가 혼잡합니다").

Photos: ReplyEngine.draft takes an optional photo path. With base.image_model set, the vision model
describes the photo and the reply model gets that description, as in the photo experiments. With
base.photo_context also true, the vision model gets the store's menu (from its documents) as a
reference, which named dishes right far more often (RESEARCH_LOG, 음식 이름). Descriptions are
not cached on disk. customer.proto has no photo field yet, so DraftReply never sends a photo:
once the backend adds one, save it to a temporary file and pass its path as image.
"""

import argparse
import hmac
import json
import re
import logging
import os
import signal
import threading
import time
import uuid
from concurrent import futures
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

from .config import Config
from .pipeline import generation_messages
from .types import Document, Review

log = logging.getLogger("scrable.serve")

SERVICE_NAME = "america.ai.v1.CustomerAssistant"
MAX_REVIEW_CHARS = 2000  # Same limit as the backend's ReplyDraftRequest.reviewText.
MAX_TONE_CHARS = 30
MAX_STORE_INFO_CHARS = 2000  # Same limit as the backend's ReplyDraftRequest.storeInfo.
OWNER_LINES = 2  # lines of the owner's store info given to the model per review
OWNER_HEADER = "가게 정보 (사장님이 등록한 내용이며, 손님이 주문한 메뉴가 아닙니다)"

# Review topics for the "keywords" field: keyword -> substrings that signal it. Simple and
# predictable on purpose; the order is the output order.
KEYWORDS = {
    "배달 지연": ("늦", "지연", "오래 걸", "넘게 걸", "시간 걸", "한참", "기다렸"),
    "음식 온도": ("식어", "식었", "식은", "미지근", "차가웠"),
    "누락": ("빠졌", "빠져", "누락", "안 왔", "안왔"),
    "이물질": ("이물질", "머리카락", "비닐"),
    "포장": ("포장", "새서", "쏟아", "흐트러"),
    "맛": ("맛있", "맛이", "맛은", "맛도", "짜", "싱거", "달아"),
    "맵기": ("맵", "매워", "순한맛"),
    "양": ("양이", "양도", "양은", "푸짐", "적어", "넉넉"),
    "가격": ("가격", "비싸", "저렴", "가성비"),
    "친절": ("친절", "불친절", "응대"),
    "요청사항": ("요청사항", "요청한", "요청했"),
    "재주문": ("또 시킬", "또 주문", "또 올", "또 갈", "재주문", "단골", "다음에 또"),
}


# Complaint and praise words, used only when the backend sends no star rating (the site does not
# send one yet). Every training input had a rating, and without one replies came out wrong (thanking
# the customer for a complaint), so a rating guessed from the text goes in instead.
NEGATIVE_CUES = (
    "늦", "식어", "식었", "미지근", "차가웠", "별로", "실망", "불친절", "아쉽", "아쉬", "최악", "불편", "불만",
    "환불", "교환", "반품", "누락", "빠졌", "빠져", "안 왔", "안왔", "이물질", "머리카락", "들떴", "들뜨",
    "새서", "쏟아", "파손", "깨져", "망가", "엉망", "답이 없", "기다렸", "길어요", "오래 걸", "비싸", "짜요",
    "싱거", "안 좋", "안좋",
)
POSITIVE_CUES = (
    "맛있", "좋아", "좋았", "좋네", "좋고", "좋은", "친절", "만족", "최고", "감사", "예쁘", "이쁘", "깨끗",
    "깔끔", "꼼꼼", "푸짐", "넉넉", "추천", "맘에 들", "마음에 들", "또 올", "또 갈", "또 시킬", "또 주문",
    "재주문", "단골", "잘 잘",
)


def estimated_rating(text):
    """Star rating guessed from a review sent without one: 2 complaint, 5 praise, 3 both (mixed
    reviews such as "맛있는데 늦었어요"), None when neither shows (the input then has no rating)."""
    negative = any(cue in text for cue in NEGATIVE_CUES)
    positive = any(cue in text for cue in POSITIVE_CUES)
    if negative and positive:
        return 3
    if negative:
        return 2
    if positive:
        return 5
    return None


def required_text(line):
    """The sentence after "필수:" on a store-info line (the owner wants it in every reply), else ""."""
    head, sep, rest = line.strip().partition(":")
    return rest.strip() if sep and head.strip() in ("필수", "[필수]") else ""


def menu_in_review(line, review_text):
    """A store-info line as given, except a menu line ("메뉴: 떡볶이, 김밥 ...") keeps only the dishes the
    review itself names, or is dropped (""). Menu names the review does not mention never reach the
    model, which otherwise took the menu for the customer's order."""
    if "메뉴" not in line.split(":")[0]:
        return line
    head, _, items = line.partition(":")
    named = [item.strip() for item in re.split(r"[,，·/]", items) if item.strip() and item.strip() in review_text]
    return f"{head.strip()}: {', '.join(named)}" if named else ""


def sentiment_of(rating):
    """positive | neutral | negative from the star rating (0 = unknown), as the backend's mock."""
    if rating >= 4:
        return "positive"
    if rating in (0, 3):
        return "neutral"
    return "negative"


def keywords_of(text, limit=5):
    return [name for name, cues in KEYWORDS.items() if any(cue in text for cue in cues)][:limit]


def store_document(store):
    """One short document from the backend's StoreContext fields, or None when all are empty."""
    labels = (("name", "가게 이름"), ("category", "업종"), ("region", "지역"), ("address", "주소"))
    lines = [f"- {label}: {store[key].strip()}" for key, label in labels if store.get(key, "").strip()]
    return "가게 기본 정보 (가게가 등록한 내용)\n" + "\n".join(lines) if lines else None


@dataclass
class ServeConfig:
    base: dict = field(default_factory=dict)  # Config fields: model, adapter, documents, prompt
    store_aliases: dict = field(default_factory=dict)  # backend store_id -> documents_dir folder
    host: str = "0.0.0.0"
    port: int = 50053
    workers: int = 8  # gRPC threads; generation itself runs one at a time
    max_waiting: int = 32  # queued DraftReply calls before RESOURCE_EXHAUSTED
    # Add the backend's tone ("- 답글 말투: ...") to the prompt. Off: no training input had one, and
    # with the site's fixed "정중하고 따뜻하게" the replies got worse (repeated phrases).
    use_tone: bool = False

    def __post_init__(self):
        self.config = Config(**self.base)
        if self.config.backend != "local":
            raise ValueError("serve runs the local model: base.backend must be local")
        if not self.config.documents_dir:
            raise ValueError("base.documents_dir is required (store documents)")
        if not isinstance(self.store_aliases, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in self.store_aliases.items()
        ):
            raise ValueError("store_aliases must map store_id strings to folder names")
        if not isinstance(self.use_tone, bool):
            raise ValueError("use_tone must be true or false")
        for name in ("port", "workers", "max_waiting"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if not k.startswith("_")})  # "_comment" keys are notes

    def to_dict(self):
        return asdict(self)


class Busy(Exception):
    """Too many calls already waiting for the GPU."""


class ReplyEngine:
    """Store lookup, retrieval and generation. One generation at a time: one model on one GPU."""

    def __init__(self, scfg, llm, retriever, prompt, describer=None, menus=None):
        self.scfg, self.llm, self.retriever, self.prompt = scfg, llm, retriever, prompt
        self.describer = describer  # PhotoDescriber without disk cache, or None (photos ignored)
        self.menus = menus or {}  # documents folder -> menu line, used when base.photo_context
        self._gpu = threading.Lock()
        self._slots = threading.BoundedSemaphore(scfg.max_waiting)

    def chunks(self, store_key, text, store, store_info=""):
        """Store documents for the reply model's retrieved_documents. Store info the owner wrote on
        the customer page comes first (chunked like data/store_docs and ranked for this review);
        otherwise the documents_dir folder of the store; otherwise none. The backend's basic store
        fields are added whenever no folder document was found."""
        found = []
        if store_info.strip():
            found = [self.owner_document(store_key, text, store_info)]
        elif store_key:
            found = self.retriever.retrieve(text, store_key, self.scfg.config.top_k)
            if found:
                return found
        info = store_document(store)
        if info:
            found = [*found, SimpleNamespace(id=f"store-{store_key or 'unknown'}", source="backend StoreContext",
                                             text=info)]
        return found

    def owner_document(self, store_key, review_text, store_info):
        """One store document from the owner's store info: only the OWNER_LINES lines most related to
        this review (BM25 + embedding), under a header saying it is store info. Given the whole text,
        the 1.5B read a menu line as the customer's order and named those dishes in the reply."""
        from .rag import rank_chunks

        lines = [part.strip(" -·•") for line in store_info.splitlines() for part in line.split(" / ")]
        lines = [kept for line in lines if line and not required_text(line)
                 and (kept := menu_in_review(line, review_text))]
        if not lines:
            lines = ["(이 리뷰와 관련된 가게 정보 없음)"]
        pieces =[Document(f"owner-{i}", line, store_key or "unknown-store", "owner") for i, line in enumerate(lines)]
        chosen = rank_chunks(review_text, pieces, getattr(self.retriever, "embedder", None), OWNER_LINES)
        chosen = sorted(chosen, key=lambda piece: int(piece.id.split("-")[1]))  # keep the owner's order
        body = "\n".join(f"- {piece.text}" for piece in chosen)
        return SimpleNamespace(id=f"owner-{store_key or 'unknown'}", source="사장님이 입력한 가게 정보",
                               text=f"{OWNER_HEADER}\n{body}")

    def describe(self, image, store_key):
        """Photo description for the reply model, or None (no photo, no vision model, or it failed:
        a reply without the photo is better than no reply). Call while holding the GPU lock."""
        if not image or self.describer is None:
            return None
        menu = self.menus.get(store_key) if self.scfg.config.photo_context else None
        try:
            return self.describer.describe(image, menu=menu)
        except Exception as error:  # Log the type only: never photo contents or paths.
            log.warning("Photo description failed (%s); replying without it", type(error).__name__)
            return None

    def draft(self, store, review_text, rating, tone="", image=None, store_info=""):
        """{'reply', 'sentiment', 'keywords'} for one review. store = StoreContext fields as a dict;
        image = path to the customer's photo, or None; store_info = store info the owner wrote."""
        if not self._slots.acquire(blocking=False):
            raise Busy
        try:
            store_id = store.get("store_id", "").strip()
            store_key = self.scfg.store_aliases.get(store_id, store_id)
            if not 1 <= rating <= 5:  # 0 = not sent: guess it from the text (see NEGATIVE_CUES)
                rating = estimated_rating(review_text) or 0
            review = Review(
                f"grpc-{uuid.uuid4().hex[:12]}",
                review_text,
                store_key or "unknown-store",
                rating=rating if 1 <= rating <= 5 else None,
            )
            prompt = self.prompt
            if self.scfg.use_tone and tone.strip():
                prompt += f"\n- 답글 말투: {tone.strip()[:MAX_TONE_CHARS]}"
            chunks = self.chunks(store_key, review_text, store, store_info)
            with self._gpu:
                description = self.describe(image, store_key)
                messages = generation_messages(prompt, review, chunks, photo_description=description)
                reply = self.llm.complete(messages, max_tokens=self.scfg.config.response_tokens).strip()
            if not reply:
                raise RuntimeError("empty reply")
            required = [text for line in store_info.splitlines() if (text := required_text(line))]
            if required:  # the owner's must-say sentences, added word for word (not left to the model)
                reply = reply + "\n\n" + " ".join(required)
            return {"reply": reply, "sentiment": sentiment_of(rating), "keywords": keywords_of(review_text)}
        finally:
            self._slots.release()


def make_servicer(engine):
    import grpc

    from .grpc_gen import customer_modules

    pb, pb_grpc = customer_modules()

    class CustomerAssistant(pb_grpc.CustomerAssistantServicer):
        def DraftReply(self, request, context):
            text = request.review_text.strip()
            if not text:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "review_text is empty")
            if len(text) > MAX_REVIEW_CHARS:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, f"review_text is over {MAX_REVIEW_CHARS} characters"
                )
            if not 0 <= request.rating <= 5:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "rating must be 0-5 (0 = unknown)")
            store = {
                k: getattr(request.store, k) for k in ("store_id", "name", "category", "region", "address")
            }
            start = time.perf_counter()
            try:
                result = engine.draft(
                    store, text, request.rating, request.tone, store_info=request.store_info[:MAX_STORE_INFO_CHARS]
                )
            except Busy:
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "too many reviews waiting; retry shortly")
            except Exception:  # Never send review text or stack traces back; log the type only.
                log.exception("DraftReply failed")
                context.abort(grpc.StatusCode.INTERNAL, "reply generation failed")
            # Log ids and timing only, never review text (it is customer data).
            log.info(
                "DraftReply store=%s rating=%s %.2fs",
                store["store_id"] or "-",
                request.rating,
                time.perf_counter() - start,
            )
            return pb.ReviewReplyResult(**result)

    return CustomerAssistant()


def internal_key_interceptor(key):
    """Reject calls without metadata x-internal-key == key. Health checks are allowed without it."""
    import grpc

    def deny(request, context):
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid internal key")

    denied = grpc.unary_unary_rpc_method_handler(deny)

    class InternalKey(grpc.ServerInterceptor):
        def intercept_service(self, continuation, details):
            if details.method.startswith("/grpc.health.v1.Health/"):
                return continuation(details)
            sent = dict(details.invocation_metadata or ()).get("x-internal-key", "")
            if hmac.compare_digest(sent.encode(), key.encode()):
                return continuation(details)
            return denied

    return InternalKey()


def build_server(servicer, key, host="0.0.0.0", port=50053, workers=8):
    """(server, bound port). Port 0 picks a free port (tests)."""
    import grpc
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc

    from .grpc_gen import customer_modules

    _, pb_grpc = customer_modules()
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers), interceptors=[internal_key_interceptor(key)]
    )
    pb_grpc.add_CustomerAssistantServicer_to_server(servicer, server)
    status = health.HealthServicer()
    for name in ("", SERVICE_NAME):  # The backend checks the default ("") service.
        status.set(name, health_pb2.HealthCheckResponse.SERVING)
    health_pb2_grpc.add_HealthServicer_to_server(status, server)
    bound = server.add_insecure_port(f"{host}:{port}")
    if not bound:
        raise RuntimeError(f"Cannot listen on {host}:{port}")
    return server, bound


def load_engine(scfg):
    from .data import load_documents
    from .experiment import load_base_prompt
    from .llm import make_backend
    from .photos import build_describer, store_menus
    from .rag import build_retriever

    config = scfg.config
    documents = load_documents(config.documents_dir, config.app_id)
    retriever = build_retriever(config, documents)
    # No CachedLLM: the server must not write customer reviews to a disk cache.
    llm = make_backend("local", config.model, config, config.response_adapter)
    describer = build_describer(config, cache=False)  # None unless base.image_model is set
    menus = store_menus(documents) if config.photo_context else {}
    return ReplyEngine(scfg, llm, retriever, load_base_prompt(config), describer, menus)


def main():
    parser = argparse.ArgumentParser(description="SCRABLE gRPC server (CustomerAssistant.DraftReply)")
    parser.add_argument("--config", default="configs/serve.json")
    parser.add_argument("--port", type=int, help="Override port (backend default 50053)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    key = os.environ.get("AI_INTERNAL_KEY", "")
    if not key:
        raise SystemExit("Set AI_INTERNAL_KEY (same value as the backend's AI_INTERNAL_KEY) before starting")
    scfg = ServeConfig.load(args.config)
    if args.port:
        scfg.port = args.port
    log.info("Loading model and store documents...")
    engine = load_engine(scfg)
    server, port = build_server(make_servicer(engine), key, scfg.host, scfg.port, scfg.workers)
    server.start()
    log.info(
        "SCRABLE gRPC server listening on %s:%s (adapter %s)", scfg.host, port, scfg.config.response_adapter
    )
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    stop.wait()
    log.info("Stopping (finishing calls in progress)...")
    server.stop(grace=30).wait()


if __name__ == "__main__":
    main()
