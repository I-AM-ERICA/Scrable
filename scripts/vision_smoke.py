"""Check that HyperCLOVAX-SEED-Vision-Instruct-3B loads and answers before a full evaluation.

    python scripts/vision_smoke.py                      # one review reply with Vision-3B and 1.5B
    python scripts/vision_smoke.py --image my_food.jpg  # also describe one image (your own photo)

Prints each reply, seconds per reply and peak GPU memory. No API calls. Use only photos you took
or may use: never real customers' photos without permission.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scrable.llm import LocalLLM, VisionLLM  # noqa: E402
from scrable.prompts import GROUNDING_RULES  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VISION = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
TEXT = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
REVIEW = "떡볶이는 맛있었는데 단무지가 하나도 안 왔어요. 다음엔 꼭 챙겨 주세요."
STORE = "- 단무지는 주문 1건당 1개 기본 제공, 추가는 요청사항에 적어 주세요.\n- 매주 월요일 휴무"


def messages(prompt_file):
    prompt = Path(prompt_file).read_text(encoding="utf-8").strip()
    payload = {
        "app_id": "bunsik-01",
        "review": REVIEW,
        "rating": 3,
        "retrieved_documents": [{"id": "store-info", "text": STORE}],
    }
    return [
        {"role": "system", "content": prompt + "\n\n" + GROUNDING_RULES},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def timed(label, function):
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    text = function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"\n[{label}] {time.perf_counter() - start:.2f} s\n{text}", flush=True)


def peak_memory():
    import torch

    if torch.cuda.is_available():
        print(f"peak GPU memory so far: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompt", default="prompts/stage1_best_prompt.txt")
    parser.add_argument("--image", help="Optional image file to describe")
    parser.add_argument("--no-compare", action="store_true", help="Skip the 1.5B comparison")
    args = parser.parse_args()
    chat = messages(args.prompt)

    vision = VisionLLM(VISION)
    peak_memory()
    vision.complete(chat, 32)  # Warm-up: the first call is slower.
    timed("Vision-3B review reply", lambda: vision.complete(chat, 256))
    if args.image:
        timed(
            "Vision-3B image",
            lambda: vision.describe_image(
                args.image, "이 사진에 보이는 음식과 상태를 한국어로 두세 문장으로 설명해 주세요."
            ),
        )
    peak_memory()
    if not args.no_compare:
        del vision
        text = LocalLLM(TEXT)
        text.complete(chat, 32)
        timed("1.5B review reply (no adapter)", lambda: text.complete(chat, 256))


if __name__ == "__main__":
    main()
