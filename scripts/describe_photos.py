"""Print Vision-3B's description of every test photo, with no judging (no API calls, 0 cost).

    python scripts/describe_photos.py                      # all photos in data/review_photos
    python scripts/describe_photos.py --prompt "다른 질문"  # try another description prompt

Use it to check the description prompt before a paid evaluation run. Descriptions made with the
default prompt are saved to the same cache the evaluation uses (configs/photo_eval_v3.json), so
the evaluation reuses them instead of describing the photos again.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scrable.llm import VisionLLM  # noqa: E402
from scrable.photos import DESCRIBE_PROMPT, PhotoDescriber  # noqa: E402
from scrable.synth_photo_reviews import find_photos  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VISION = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--photos", default="data/review_photos")
    parser.add_argument("--prompt", default=DESCRIBE_PROMPT)
    parser.add_argument("--cache", default="outputs/photo_eval_v3/with_desc/photo_cache")
    args = parser.parse_args()

    describer = PhotoDescriber(VisionLLM(VISION), args.cache, prompt=args.prompt)
    print(f"prompt: {args.prompt}\n", flush=True)
    empty = 0
    for store, path in find_photos(args.photos):
        start = time.perf_counter()
        text = describer.describe(path)
        seconds = time.perf_counter() - start
        empty += len(text) < 15
        print(f"[{store}/{path.name}] {seconds:.1f} s\n{text}\n", flush=True)
    print(f"{empty} descriptions shorter than 15 characters (likely empty or copied)", flush=True)


if __name__ == "__main__":
    main()
