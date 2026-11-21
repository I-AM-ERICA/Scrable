"""Photo reviews: Vision-3B describes the customer's photo, the review model answers with it (plan A).

A review row may carry "image": a path to the customer's photo. With Config.image_model set, the
photo is described once (cached by image content) and the description goes into the generator
input as "photo_description". The API judge sees the photo itself, so a wrong description is
caught as a wrong reply. Use only photos you may use: never real customers' photos without consent.
"""

import base64
import hashlib
import io
import json
from pathlib import Path

from .io import write_json

MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
MAX_SIDE = 1024  # Long side in pixels sent to the API judge; smaller images cost fewer tokens.

# Prompt history (keep it short for a 3B model):
# v1 asked about amount, packaging and problems one by one -> every item answered with hedging
#    ("포장 상태는 사진으로 확인할 수 없으나...") and guesses.
# v2 a two-line form with defaults ("확실하지 않으면 '알 수 없음'", "없으면 '없음'") -> 15 of 16
#    photos came back as just "알 수 없음, 없음": the model copied the defaults.
# v3 one plain question, no form and no default values to copy.
DESCRIBE_PROMPT = (
    "이 사진 속 음식만 한국어 한두 문장으로 설명해 주세요. "
    "음식 이름과 모양을 말하고, 이물질이나 쏟아진 것처럼 눈에 띄는 문제가 보이면 함께 말해 주세요. "
    "사진에 보이지 않는 맛이나 온도는 쓰지 마세요."
)
# With a known dish name (labelled training photos). Without the name, Vision-3B misnamed dishes
# (갈치구이 -> 고등어, 계란찜 -> 짜장면) and invented taste and recipe details.
HINT_PROMPT = (
    "사진 속 음식은 {food}입니다. 사진에 보이는 모양, 양, 담긴 그릇만 한두 문장으로 짧게 설명해 주세요. "
    "이물질이나 쏟아진 것처럼 눈에 띄는 문제가 보이면 함께 말해 주세요. 맛, 온도, 조리법은 쓰지 마세요."
)
# At serving time the dish name is unknown, but the store's menu and the customer's review are:
# naming the dish becomes picking from the menu (Config.photo_context).
MAX_HINT_CHARS = 300


def context_prompt(review=None, menu=None):
    """Description question with the store's menu and the review as hints (either may be None)."""
    lines = ["이 사진은 손님이 리뷰와 함께 올린 사진입니다."]
    if menu:
        lines.append(f"가게 메뉴: {menu[:MAX_HINT_CHARS]}")
    if review:
        lines.append(f"손님 리뷰: {review[:MAX_HINT_CHARS]}")
    lines.append(
        "사진 속 음식이 메뉴 중 무엇인지 먼저 쓰고, 사진에 보이는 모양과 양, 이물질이나 쏟아진 것 같은 "
        "문제가 있으면 한두 문장으로 설명해 주세요. 리뷰는 참고만 하고 사진에 실제로 보이는 것만 쓰세요. "
        "맛, 온도, 조리법은 쓰지 마세요."
    )
    return "\n".join(lines)


def menu_text(document_text):
    """The '## 메뉴' section of a store document as one line, or '' when there is none."""
    items, inside = [], False
    for line in document_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            inside = "메뉴" in stripped
            continue
        if inside and stripped:
            items.append(stripped.lstrip("-* ").strip())
    return " / ".join(items)


def store_menus(documents):
    """{app_id: menu line} from store documents (several documents of one store are joined)."""
    menus = {}
    for doc in documents:
        text = menu_text(doc.text)
        if text:
            menus[doc.app_id] = f"{menus[doc.app_id]} / {text}" if doc.app_id in menus else text
    return menus


def check_image(path):
    path = Path(path)
    if path.suffix.lower() not in MEDIA_TYPES:
        raise ValueError(f"Unsupported image type {path.suffix}: use {sorted(MEDIA_TYPES)}")
    if not path.is_file():
        raise ValueError(f"Image not found: {path}")
    return path


def media_type_of(data, path):
    """Media type from the file's first bytes; the extension can lie (AI Hub has PNGs named .jpg,
    which the API rejects as a media-type mismatch). Falls back to the extension."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return MEDIA_TYPES[Path(path).suffix.lower()]


def image_bytes(path, max_side=MAX_SIDE):
    """(bytes, media type), downscaled so the long side is at most max_side when Pillow is present."""
    path = check_image(path)
    data = path.read_bytes()
    try:
        from PIL import Image
    except ImportError:
        return data, media_type_of(data, path)
    with Image.open(io.BytesIO(data)) as image:
        if max(image.size) <= max_side:
            return data, media_type_of(data, path)
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=85)
        return out.getvalue(), "image/jpeg"


def image_block(path):
    """Anthropic Messages API image content block for a local photo."""
    data, media_type = image_bytes(path)
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": base64.b64encode(data).decode()},
    }


class PhotoDescriber:
    """Cached photo -> Korean description with a vision model (anything with describe_image)."""

    def __init__(self, vision_llm, directory, prompt=DESCRIBE_PROMPT, max_tokens=128):
        self.vision = vision_llm
        self.directory = Path(directory)
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.calls = 0

    def key(self, path, prompt=None):
        content = hashlib.sha256(check_image(path).read_bytes()).hexdigest()
        identity = json.dumps(getattr(self.vision, "identity", {}), sort_keys=True, default=str)
        prompt = prompt or self.prompt
        return hashlib.sha256(f"{content}|{prompt}|{self.max_tokens}|{identity}".encode()).hexdigest()

    def describe(self, path, food=None, review=None, menu=None):
        """Description of the photo. food: the dish name when it is known (labelled training
        photos), so the 3B model describes the right dish instead of guessing it. review / menu:
        the customer's review and the store's menu as hints when the dish is not known."""
        if food:
            prompt = HINT_PROMPT.format(food=food)
        elif review or menu:
            prompt = context_prompt(review, menu)
        else:
            prompt = self.prompt
        cache = self.directory / f"{self.key(path, prompt)}.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))["text"]
        text = self.vision.describe_image(path, prompt, self.max_tokens).strip()
        if not text:
            raise ValueError(f"Empty photo description for {path}")
        self.calls += 1
        write_json(cache, {"text": text, "image": str(path)})
        return text


def build_describer(config):
    """PhotoDescriber for Config.image_model, or None when photos are not described."""
    if not config.image_model:
        return None
    from .llm import make_backend

    vision = make_backend("local", config.image_model, config)
    return PhotoDescriber(vision, Path(config.output_dir) / "photo_cache")
