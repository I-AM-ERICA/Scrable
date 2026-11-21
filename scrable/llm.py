"""Interchangeable real LLM backends, persistent request cache and JSON validation."""

import json
import re
import threading
from pathlib import Path

from .io import digest, write_json

HYPERCLOVA_STOPS = ("<|endofturn|>", "<|stop|>")


def is_hyperclova(model_name):
    return "hyperclovax" in model_name.lower()


def chat_input(model_name, messages):
    """HyperCLOVA X SEED expects an (empty) tool_list turn before the conversation."""
    return [{"role": "tool_list", "content": ""}, *messages] if is_hyperclova(model_name) else list(messages)


def completion_end(model_name, tokenizer):
    """Text that ends an assistant turn, as the model itself emits it (used as a training target)."""
    return "<|im_end|><|endofturn|>" if is_hyperclova(model_name) else tokenizer.eos_token


def file_digest(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapter_identity(adapter):
    files = sorted(p for p in Path(adapter).glob("adapter_*") if p.is_file())
    if not files:
        raise ValueError(f"No LoRA adapter files (adapter_config.json, adapter_model.*) in {adapter}")
    return {"path": str(adapter), "sha256": {p.name: file_digest(p) for p in files}}


class LocalLLM:
    def __init__(self, model, device="auto", context_tokens=8192, seed=42, adapter=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        set_seed(seed)
        self.model_name = model
        if device == "auto":
            device = (
                "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = device
        self.context_tokens = context_tokens
        dtype = (
            torch.bfloat16
            if device.startswith("cuda") and torch.cuda.is_bf16_supported()
            else torch.float16
            if device.startswith("cuda") or device == "mps"
            else torch.float32
        )
        print(f"Loading {model} on {device} (CPU can be very slow)...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype,
            trust_remote_code=False,
        )
        if adapter:
            from peft import PeftModel

            print(f"Applying LoRA adapter {adapter}", flush=True)
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        self.model = self.model.to(device).eval()
        self.identity = {
            "backend": "transformers",
            "model": model,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "device": device,
            "context_tokens": context_tokens,
            "seed": seed,
            "decode": "greedy-v1",
        }
        if adapter:  # Absent without an adapter, so stage-1 cache keys stay valid.
            self.identity["adapter"] = adapter_identity(adapter)

    def _encode(self, messages, max_tokens):
        text = self.tokenizer.apply_chat_template(
            chat_input(self.model_name, messages), tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        size = inputs["input_ids"].shape[1]
        limit = min(
            self.context_tokens, getattr(self.model.config, "max_position_embeddings", self.context_tokens)
        )
        if size + max_tokens > limit:
            raise ValueError(
                f"LLM context overflow ({size}+{max_tokens}>{limit}). "
                "Reduce train_size/top_k/chunk_tokens or increase context_tokens."
            )
        stop_kwargs = (
            {"stop_strings": list(HYPERCLOVA_STOPS), "tokenizer": self.tokenizer}
            if is_hyperclova(self.model_name)
            else {}
        )
        return inputs, size, stop_kwargs

    def complete(self, messages, max_tokens):
        import torch

        inputs, size, stop_kwargs = self._encode(messages, max_tokens)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                **stop_kwargs,
            )
        return self.tokenizer.decode(output[0, size:], skip_special_tokens=True).strip()

    def sample(self, messages, max_tokens, n, temperature, top_p, seed):
        """n sampled answers in one batch (stage-2 candidates). Reproducible for a fixed seed."""
        import torch
        from transformers import set_seed

        inputs, size, stop_kwargs = self._encode(messages, max_tokens)
        set_seed(seed)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.eos_token_id,
                **stop_kwargs,
            )
        texts = []
        for row in output[:, size:]:
            text = self.tokenizer.decode(row, skip_special_tokens=False)
            for marker in ("<|im_end|>", *HYPERCLOVA_STOPS, self.tokenizer.eos_token or "<|endoftext|>"):
                text = text.split(marker)[0]
            texts.append(text.strip())
        return texts


def is_vision(model_name):
    """HyperCLOVAX-SEED-Vision-Instruct models need their own loader (remote code, processor)."""
    return is_hyperclova(model_name) and "vision" in model_name.lower()


def is_image_text(model_name):
    """Pretrained image-text-to-text models other than HyperCLOVAX Vision (e.g. Qwen2.5-VL)."""
    return "-vl" in model_name.lower() and not is_vision(model_name)


def vision_messages(messages):
    """Our {"role", "content": str} chat as the Vision model's typed content list."""
    return [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]


def clean_completion(text, eos_token=None):
    """Answer text up to the first end-of-turn marker."""
    for marker in ("<|im_end|>", *HYPERCLOVA_STOPS, eos_token or "<|endoftext|>"):
        text = text.split(marker)[0]
    return text.strip()


def new_tokens(output_row, input_ids):
    """Generated tokens only. Some remote-code models return the prompt too, some do not."""
    size = input_ids.shape[-1]
    if output_row.shape[-1] > size and output_row[:size].tolist() == input_ids[0].tolist():
        return output_row[size:]
    return output_row


class VisionLLM:
    """HyperCLOVAX-SEED-Vision-Instruct-3B as a text generator (same interface as LocalLLM).

    Loads NAVER's remote code (trust_remote_code=True) and needs `pip install einops timm av decord`.
    Text-only for now; describe_image() is for checking image input by hand. No LoRA adapter yet.
    """

    def __init__(self, model, device="auto", context_tokens=8192, seed=42):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, set_seed

        set_seed(seed)
        self.model_name = model
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.context_tokens = context_tokens
        dtype = (
            torch.bfloat16
            if device.startswith("cuda") and torch.cuda.is_bf16_supported()
            else torch.float16
            if device.startswith("cuda")
            else torch.float32
        )
        print(f"Loading {model} (remote code) on {device}...", flush=True)
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype, trust_remote_code=True)
            .to(device)
            .eval()
        )
        self.identity = {
            "backend": "transformers-vision",
            "model": model,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "device": device,
            "context_tokens": context_tokens,
            "seed": seed,
            "decode": "greedy-v1",
        }

    def _encode(self, chat, max_tokens):
        inputs = self.processor.apply_chat_template(
            chat, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True
        ).to(self.device)
        size = inputs["input_ids"].shape[-1]
        if size + max_tokens > self.context_tokens:
            raise ValueError(f"LLM context overflow ({size}+{max_tokens}>{self.context_tokens}).")
        return inputs

    def _generate(self, inputs, **kwargs):
        import torch

        stops = {"stop_strings": list(HYPERCLOVA_STOPS), "tokenizer": self.tokenizer}
        with torch.inference_mode():
            try:
                return self.model.generate(
                    **inputs, pad_token_id=self.tokenizer.eos_token_id, **stops, **kwargs
                )
            except (TypeError, ValueError):  # Remote generate() without stop_strings support.
                return self.model.generate(**inputs, pad_token_id=self.tokenizer.eos_token_id, **kwargs)

    def _texts(self, output, inputs):
        return [
            clean_completion(
                self.tokenizer.decode(new_tokens(row, inputs["input_ids"]), skip_special_tokens=False),
                self.tokenizer.eos_token,
            )
            for row in output
        ]

    def complete(self, messages, max_tokens):
        inputs = self._encode(vision_messages(messages), max_tokens)
        return self._texts(self._generate(inputs, max_new_tokens=max_tokens, do_sample=False), inputs)[0]

    def sample(self, messages, max_tokens, n, temperature, top_p, seed):
        from transformers import set_seed

        inputs = self._encode(vision_messages(messages), max_tokens)
        set_seed(seed)
        output = self._generate(
            inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n,
        )
        return self._texts(output, inputs)

    def describe_image(self, image_path, question, max_tokens=256):
        """One image + question -> text. For a manual check of image input (scripts/vision_smoke.py)."""
        chat = [
            {
                "role": "user",
                "content": [{"type": "image", "image": str(image_path), "filename": Path(image_path).name}],
            },
            {"role": "user", "content": [{"type": "text", "text": question}]},
        ]
        inputs = self._encode(chat, max_tokens)
        return self._texts(self._generate(inputs, max_new_tokens=max_tokens, do_sample=False), inputs)[0]


class ImageTextLLM:
    """A pretrained Hugging Face image-text-to-text model used only to describe photos, e.g.
    Qwen/Qwen2.5-VL-7B-Instruct (Apache-2.0). Same describe_image() as VisionLLM.

    Loaded with AutoModelForImageTextToText and the model's own processor; no remote code.
    max_pixels caps the image size for processors that accept it (Qwen2-VL family).
    """

    def __init__(self, model, device="auto", seed=42, max_pixels=1024 * 28 * 28):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

        set_seed(seed)
        self.model_name = model
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        dtype = (
            torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
        )
        print(f"Loading {model} (image-text-to-text) on {device}...", flush=True)
        try:
            self.processor = AutoProcessor.from_pretrained(model, max_pixels=max_pixels)
        except (TypeError, ValueError):
            self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForImageTextToText.from_pretrained(model, torch_dtype=dtype).to(device).eval()
        self.identity = {
            "backend": "transformers-image-text",
            "model": model,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "device": device,
            "max_pixels": max_pixels,
            "seed": seed,
            "decode": "greedy-v1",
        }

    @staticmethod
    def _load(path):
        from PIL import Image

        with Image.open(path) as source:
            return source.convert("RGB")

    def _run(self, chat, images, max_tokens):
        import torch

        text = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        new = output[0, inputs["input_ids"].shape[-1] :]
        return self.processor.decode(new, skip_special_tokens=True).strip()

    def describe_image(self, image_path, question, max_tokens=256):
        chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        return self._run(chat, [self._load(image_path)], max_tokens)

    def complete(self, messages, max_tokens):
        """Our chat as the reply generator. A user message may be a list of parts:
        {"type": "image", "path": ...} and {"type": "text", "text": ...} (see generation_messages)."""
        chat, images = [], []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                parts = [{"type": "text", "text": content}]
            else:
                parts = []
                for part in content:
                    if part["type"] == "image":
                        images.append(self._load(part["path"]))
                        parts.append({"type": "image"})
                    else:
                        parts.append({"type": "text", "text": part["text"]})
            chat.append({"role": message["role"], "content": parts})
        return self._run(chat, images, max_tokens)


class OpenAILLM:
    def __init__(self, model, min_output_tokens=1):
        from openai import OpenAI

        self.client = OpenAI(timeout=180, max_retries=3)
        self.model_name = model
        # Reasoning models count hidden reasoning in max_completion_tokens; budgets act as a floor.
        self.min_output_tokens = min_output_tokens
        self.identity = {
            "backend": "openai",
            "model": model,
            "base_url": str(self.client.base_url),
            "min_output_tokens": min_output_tokens,
            "request_version": 2,
        }

    def complete(self, messages, max_tokens):
        result = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=max(max_tokens, self.min_output_tokens),
        )
        message = result.choices[0].message
        if message.refusal:
            raise RuntimeError(f"Model refused request: {message.refusal}")
        return message.content or ""


class CachedLLM:
    def __init__(self, backend, directory):
        self.backend = backend
        self.directory = Path(directory)
        self.calls = 0
        self.hits = 0
        self._lock = threading.Lock()  # Counters stay correct when an API judge runs in threads.

    def complete(self, messages, max_tokens):
        key = digest({"backend": self.backend.identity, "messages": messages, "max_tokens": max_tokens})
        path = self.directory / f"{key}.json"
        if path.exists():
            with self._lock:
                self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        text = self.backend.complete(messages, max_tokens)
        with self._lock:
            self.calls += 1
        if not isinstance(text, str) or not text.strip():
            raise ValueError("LLM returned an empty response. Check output token budget/model configuration.")
        write_json(path, {"text": text, "backend": self.backend.identity})
        return text

    def sample(self, messages, max_tokens, n, temperature, top_p, seed):
        """Cached batch of sampled answers (local backends only). Empty answers are kept as ""."""
        sampling = {"n": n, "temperature": temperature, "top_p": top_p, "seed": seed}
        key = digest(
            {
                "backend": self.backend.identity,
                "messages": messages,
                "max_tokens": max_tokens,
                "sampling": sampling,
            }
        )
        path = self.directory / f"{key}.json"
        if path.exists():
            with self._lock:
                self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["texts"]
        texts = self.backend.sample(messages, max_tokens, n, temperature, top_p, seed)
        with self._lock:
            self.calls += 1
        if not isinstance(texts, list) or len(texts) != n or not all(isinstance(t, str) for t in texts):
            raise ValueError(f"Sampling must return {n} strings")
        write_json(path, {"texts": texts, "sampling": sampling, "backend": self.backend.identity})
        return texts


def parse_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def structured_call(llm, messages, validator, max_tokens, attempts=3):
    conversation = [dict(m) for m in messages]
    for attempt in range(attempts):
        raw = llm.complete(conversation, max_tokens=max_tokens)
        try:
            return validator(parse_json_object(raw))
        except (ValueError, TypeError, KeyError) as exc:
            if attempt + 1 == attempts:
                raise ValueError(
                    f"Invalid structured LLM output after {attempts} attempts: {exc}. "
                    "Use a stronger model or increase the output token budget."
                ) from exc
            conversation.extend(
                [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"Invalid JSON/schema: {exc}. Return only the corrected JSON object. "
                        "Use numeric scores from 1 to 5 and the exact requested keys.",
                    },
                ]
            )


def make_backend(backend, model, config, adapter=None):
    if adapter and backend != "local":
        raise ValueError("A LoRA adapter can only be applied to a local model")
    if backend == "local" and is_vision(model):
        if adapter:
            raise ValueError("LoRA adapters on the Vision model are not supported yet")
        return VisionLLM(model, config.device, config.context_tokens, config.seed)
    if backend == "local" and is_image_text(model):
        if adapter:
            raise ValueError("LoRA adapters on image-text models are not supported yet")
        return ImageTextLLM(model, config.device, config.seed)
    if backend == "local":
        return LocalLLM(model, config.device, config.context_tokens, config.seed, adapter)
    if backend == "openai":
        return OpenAILLM(model, config.api_min_output_tokens)
    if backend == "anthropic":
        from .anthropic_llm import AnthropicLLM

        return AnthropicLLM(model, config.api_min_output_tokens, config.anthropic_effort)
    raise ValueError(f"Unknown backend: {backend}")


def build_llms(config, specs=None, shared=None):
    """Build cached LLMs for (backend, model) specs; identical specs share one model object.

    Default specs are the response, judge and prompt roles, so a small local generator can be
    taught by a stronger API judge/prompt generator. A spec may carry a third item, a LoRA adapter
    path, so the trained generator never shares an object with a judge on the same base model.
    """
    if specs is None:
        specs = [response_spec(config), config.role("judge"), config.role("prompt")]
    shared = {} if shared is None else shared
    for spec in dict.fromkeys(specs):
        if spec not in shared:
            backend = make_backend(spec[0], spec[1], config, *spec[2:])
            shared[spec] = CachedLLM(backend, Path(config.output_dir) / "llm_cache")
    return tuple(shared[spec] for spec in specs)


def response_spec(config):
    """(backend, model) of the response generator, plus the LoRA adapter path when one is set."""
    spec = config.role("response")
    return (*spec, config.response_adapter) if config.response_adapter else spec
