"""Call the SCRABLE gRPC server the way the backend does: health check, then one DraftReply.

    read -rs AI_INTERNAL_KEY && export AI_INTERNAL_KEY
    python scripts/grpc_client.py                                   # localhost:50053, sample review
    python scripts/grpc_client.py --host <서버IP> --store-id bunsik-01 --rating 2 --review "배달이 늦었어요"

Prints the health status, the reply, sentiment, keywords and the round-trip time.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scrable.grpc_gen import customer_modules  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50053)
    parser.add_argument("--store-id", default="bunsik-01")
    parser.add_argument("--store-name", default="예시분식")
    parser.add_argument("--category", default="분식")
    parser.add_argument("--rating", type=int, default=2)
    parser.add_argument("--tone", default="")
    parser.add_argument("--store-info", default="", help="Store info the owner wrote (menu, hours, delivery notes)")
    parser.add_argument(
        "--review", default="떡볶이는 맛있었는데 배달이 한 시간 넘게 걸려서 다 식어서 왔어요."
    )
    args = parser.parse_args()

    import grpc
    from grpc_health.v1 import health_pb2, health_pb2_grpc

    pb, pb_grpc = customer_modules()
    from america.ai.v1 import common_pb2  # generated next to customer_pb2

    key = os.environ.get("AI_INTERNAL_KEY", "")
    if not key:
        raise SystemExit("Set AI_INTERNAL_KEY first (same value as the server)")
    with grpc.insecure_channel(f"{args.host}:{args.port}") as channel:
        health = health_pb2_grpc.HealthStub(channel).Check(health_pb2.HealthCheckRequest(), timeout=5)
        print(f"health: {health_pb2.HealthCheckResponse.ServingStatus.Name(health.status)}")
        request = pb.ReviewReplyRequest(
            store=common_pb2.StoreContext(
                store_id=args.store_id, name=args.store_name, category=args.category
            ),
            review_text=args.review,
            rating=args.rating,
            tone=args.tone,
            store_info=args.store_info,
        )
        start = time.perf_counter()
        result = pb_grpc.CustomerAssistantStub(channel).DraftReply(
            request, metadata=(("x-internal-key", key),), timeout=90
        )
        print(f"seconds: {time.perf_counter() - start:.2f}")
        print(f"sentiment: {result.sentiment}\nkeywords: {list(result.keywords)}\nreply:\n{result.reply}")


if __name__ == "__main__":
    main()
