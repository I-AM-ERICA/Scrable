"""Interchangeable real LLM backends, persistent request cache and JSON validation."""

import json
import re
from pathlib import Path

from .io import digest, write_json


class LocalLLM:
    def __init__(self, model, device="auto", context_tokens=8192, seed=42):
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
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=dtype,
                trust_remote_code=False,
            )
            .to(device)
            .eval()
        )
        self.identity = {
            "backend": "transformers",
            "model": model,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "device": device,
            "context_tokens": context_tokens,
            "seed": seed,
            "decode": "greedy-v1",
        }

    def complete(self, messages, max_tokens):
        import torch

        is_hyperclova = "hyperclovax" in self.model_name.lower()
        if is_hyperclova:
            messages = [{"role": "tool_list", "content": ""}, *messages]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
            {"stop_strings": ["<|endofturn|>", "<|stop|>"], "tokenizer": self.tokenizer}
            if is_hyperclova
            else {}
        )
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                **stop_kwargs,
            )
        return self.tokenizer.decode(output[0, size:], skip_special_tokens=True).strip()


class OpenAILLM:
    def __init__(self, model):
        from openai import OpenAI

        self.client = OpenAI(timeout=90, max_retries=3)
        self.model_name = model
        self.identity = {
            "backend": "openai",
            "model": model,
            "base_url": str(self.client.base_url),
            "request_version": 1,
        }

    def complete(self, messages, max_tokens):
        result = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
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

    def complete(self, messages, max_tokens):
        key = digest({"backend": self.backend.identity, "messages": messages, "max_tokens": max_tokens})
        path = self.directory / f"{key}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        text = self.backend.complete(messages, max_tokens)
        self.calls += 1
        if not isinstance(text, str) or not text.strip():
            raise ValueError("LLM returned an empty response. Check output token budget/model configuration.")
        write_json(path, {"text": text, "backend": self.backend.identity})
        return text


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


def build_llms(config):
    models = [config.model, config.judge_model or config.model, config.prompt_model or config.model]
    shared = {}
    for model in dict.fromkeys(models):
        backend = (
            LocalLLM(model, config.device, config.context_tokens, config.seed)
            if config.backend == "local"
            else OpenAILLM(model)
        )
        shared[model] = CachedLLM(backend, Path(config.output_dir) / "llm_cache")
    return tuple(shared[m] for m in models)
